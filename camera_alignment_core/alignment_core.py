import logging
from typing import Tuple

from aicsimageio import AICSImage
import numpy
import numpy.typing
import skimage
import skimage.transform

from .alignment_utils import (
    AlignmentInfo,
    CropRings,
    RingAlignment,
    SegmentRings,
    get_center_z,
)
from .channel_info import ChannelInfo
from .constants import (
    LOGGER_NAME,
    Channel,
    Magnification,
)
from .exception import (
    IncompatibleImageException,
    UnsupportedMagnification,
)

log = logging.getLogger(LOGGER_NAME)


class AlignmentCore:
    """Wrapper for core of camera alignment algorithm"""

    def generate_alignment_matrix(
        self,
        optical_control_image: numpy.typing.NDArray[numpy.uint16],
        reference_channel: int,
        shift_channel: int,
        magnification: int,
        px_size_xy: float,
    ) -> Tuple[numpy.typing.NDArray[numpy.float16], AlignmentInfo]:

        log.debug(
            "Params -- reference_channel: %s; shift_channel: %s; magnification: %s; px_size_xy: %s",
            reference_channel,
            shift_channel,
            magnification,
            px_size_xy,
        )

        if not optical_control_image.ndim == 4:
            raise IncompatibleImageException(
                f"Expected optical_control_image to be 4 dimensional ('CZYX'). Got: {optical_control_image.shape}"
            )

        # detect center z-slice on reference channel
        log.debug("detecing center z in ref")
        ref_center_z = get_center_z(
            img_stack=optical_control_image[reference_channel, :, :, :]
        )

        # Crop with all available rings
        log.debug("crop rings")
        ref_crop, crop_dims = CropRings(
            img=optical_control_image[reference_channel, ref_center_z, :, :],
            pixel_size_um=px_size_xy,
            magnification=magnification,
            filter_px_size=50,
        ).run()

        mov_crop = optical_control_image[
            shift_channel,
            ref_center_z,
            crop_dims[0] : crop_dims[1],
            crop_dims[2] : crop_dims[3],
        ]

        # segment rings on reference image
        log.debug("segment rings in ref")
        (
            ref_seg_rings,
            ref_seg_rings_label,
            ref_props_df,
            ref_cross_label,
        ) = SegmentRings(ref_crop, px_size_xy, magnification, thresh=None).run()

        # segment rings on moving image
        log.debug("segment rings in moving")
        (
            mov_seg_rings,
            mov_seg_rings_label,
            mov_props_df,
            mov_cross_label,
        ) = SegmentRings(mov_crop, px_size_xy, magnification, thresh=None).run()

        # Create alignment from segmentation
        log.debug("Creating alignment matrix")
        similarity_transform, align_info = RingAlignment(
            ref_seg_rings,
            ref_seg_rings_label,
            ref_props_df,
            ref_cross_label,
            mov_seg_rings,
            mov_seg_rings_label,
            mov_props_df,
            mov_cross_label,
        ).run()

        return similarity_transform.params, align_info

    def align_image(
        self,
        alignment_matrix: numpy.typing.NDArray[numpy.float16],
        image: numpy.typing.NDArray[numpy.uint16],
        channel_info: ChannelInfo,
        magnification: int,
        crop: bool = True,
    ) -> numpy.typing.NDArray[numpy.uint16]:
        """
        Align a CZYX `image` using `alignment_matrix`.
        Uses `channel_info` to know which channels within `image` to align.
        Uses `magnification` to know how to crop the resulting aligned image.
        Does not crop aligned image if crop=False.
        """
        if not image.ndim == 4:
            raise IncompatibleImageException(
                f"Expected image to be 4 dimensional ('CZYX'). Got: {image.shape}"
            )

        if magnification not in [
            supported_magnification.value
            for supported_magnification in list(Magnification)
        ]:
            raise UnsupportedMagnification(
                f"Cannot perform image alignment for magnification {str(magnification)}."
            )

        if not channel_info:
            raise ValueError(
                "Passed an empty channel_info `align_image`. Cannot determine which channels to align."
            )

        # If no channel within the image is known to require alignment, fail.
        if not any([channel.requires_alignment() for channel, _ in channel_info]):
            raise IncompatibleImageException(
                f"No channels within image require alignment. Channel info: {channel_info}"
            )

        # Build up the aligned image by iterating over the input image and aligning the channels
        # that require alignment
        aligned_image = numpy.empty(image.shape, dtype=numpy.uint16)
        channels, *_ = image.shape
        for index in range(0, channels):
            try:
                channel = channel_info.channel_at_index(index)
            except IndexError:
                log.warning("Encountered an unknown channel at index %s", index)
                aligned_image[index] = image[index]
            else:
                if channel.requires_alignment():
                    log.debug("Applying alignment to %s channel", channel.value)
                    aligned_slice = self._apply_alignment_matrix(
                        alignment_matrix, image[index]
                    )
                    aligned_image[index] = aligned_slice
                else:
                    log.debug("Skipping alignment for %s channel", channel.value)
                    aligned_image[index] = image[index]

        if crop:
            # Some of the channels were aligned. Crop to adjust for differences in border position.
            return self._crop(aligned_image, Magnification(magnification))

        return aligned_image

    def get_channel_info(self, image: AICSImage) -> ChannelInfo:
        """
        Map channel names to their corresponding index position within image data. If an unknown channel name
        is encountered, a warning is logged.
        """
        channels = image.channel_names
        channel_info_dict = dict()
        for channel in channels:
            if channel in ["Bright", "Bright_2", "TL_100x"]:
                channel_info_dict.update(
                    {Channel.RAW_BRIGHTFIELD: channels.index(channel)}
                )
            elif channel in ["EGFP"]:
                channel_info_dict.update({Channel.RAW_488_NM: channels.index(channel)})
            elif channel in ["CMDRP"]:
                channel_info_dict.update({Channel.RAW_638_NM: channels.index(channel)})
            elif channel in ["H3342"]:
                channel_info_dict.update({Channel.RAW_405_NM: channels.index(channel)})
            elif channel in ["TaRFP", "TagRFP"]:
                channel_info_dict.update({Channel.RAW_561_NM: channels.index(channel)})
            else:
                log.warning("Encountered unknown channel: %s", channel)

        return ChannelInfo(channel_info_dict)

    def _crop(
        self,
        image: numpy.typing.NDArray[numpy.uint16],
        magnification: Magnification,
        black_pixel_cutoff: int = 50,
    ) -> numpy.typing.NDArray[numpy.uint16]:
        """Crops a CZYX image based on the magnification used to generate the image"""

        if not image.ndim == 4:
            raise IncompatibleImageException(
                f"Expected image to be 4 dimensional ('CZYX'). Got: {image.shape}"
            )

        cropping_dimension = magnification.cropping_dimension
        log.debug(
            "Cropping to <X: %s, Y: %s>", cropping_dimension.x, cropping_dimension.y
        )

        (_, _, Y, X) = image.shape

        assert (
            Y >= cropping_dimension.y
        ), f"Image is smaller than intended cropping dimension in Y: actual == {Y}; intended == {cropping_dimension.y}"
        assert (
            X >= cropping_dimension.x
        ), f"Image is smaller than intended cropping dimension in X: actual == {X}; intended == {cropping_dimension.x}"

        half_diff_x = (X - cropping_dimension.x) // 2
        half_diff_y = (Y - cropping_dimension.y) // 2
        cropped_image: numpy.typing.NDArray[numpy.uint16] = image[
            :,  # C
            :,  # Z
            half_diff_y : half_diff_y + cropping_dimension.y,  # Y
            half_diff_x : half_diff_x + cropping_dimension.x,  # X
        ]

        # Check if there are black pixels, if so, crop a little further
        if numpy.any(cropped_image < black_pixel_cutoff):
            SMALL_AMOUNT_OF_ADDITIONAL_CROP = 20  # px
            log.warning(
                "After cropping, detected pixels under %s. Taking off an additional %s pixels",
                black_pixel_cutoff,
                SMALL_AMOUNT_OF_ADDITIONAL_CROP,
            )

            (_, _, Y, X) = cropped_image.shape
            out_x = cropping_dimension.x - SMALL_AMOUNT_OF_ADDITIONAL_CROP
            out_y = cropping_dimension.y - SMALL_AMOUNT_OF_ADDITIONAL_CROP
            half_diff_x = (X - out_x) // 2
            half_diff_y = (Y - out_y) // 2
            cropped_image = cropped_image[
                :,  # C
                :,  # Z
                half_diff_y : half_diff_y + out_y,  # Y
                half_diff_x : half_diff_x + out_x,  # X
            ]

        assert not numpy.any(
            cropped_image < black_pixel_cutoff
        ), "Black pixels are detected, either from original image or due to alignment"

        return cropped_image

    def _apply_alignment_matrix(
        self,
        alignment_matrix: numpy.typing.NDArray[numpy.float16],
        image_slice: numpy.typing.NDArray[numpy.uint16],
    ) -> numpy.typing.NDArray[numpy.uint16]:
        """
        Applies an affine transformation matrix to a 3D (ZYX) slice of a multi-channel image.
        """
        if not image_slice.ndim == 3:
            raise IncompatibleImageException(
                f"Cannot perform similarity matrix transform: invalid image dimensions. \
                Image must be 3D but detected {len(image_slice.shape)} dimensions"
            )

        after_transform = numpy.empty(image_slice.shape, dtype=numpy.double)
        for z in range(0, after_transform.shape[0]):
            after_transform[z, :, :] = skimage.transform.warp(
                image_slice[z, :, :],
                inverse_map=alignment_matrix,
                order=3,
            )

        # skimage.transform.warp converts input image to the float range (0..1), so rescale to uint16
        return (after_transform * numpy.iinfo(numpy.uint16).max).astype(numpy.uint16)
