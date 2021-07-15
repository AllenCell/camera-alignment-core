import logging
from typing import Dict, List, Tuple

from aicsimageio import AICSImage
import numpy
import numpy.typing
from skimage import transform

from .alignment_utils import (
    AlignmentInfo,
    CropRings,
    RingAlignment,
    SegmentRings,
    get_center_z,
)
from .constants import LOGGER_NAME

from .exception import IncompatibleImageException

log = logging.getLogger(LOGGER_NAME)

from skimage import transform


class AlignmentCore:
    """Wrapper for core of camera alignment algorithm"""

    def generate_alignment_matrix(
        self,
        optical_control_image: numpy.typing.NDArray[numpy.uint16],
        reference_channel: int,
        shift_channel: int,
        magnification: int,
        px_size_xy: float,
    ) -> Tuple[numpy.typing.NDArray[numpy.uint16], AlignmentInfo]:

        # if more than 4 dimensions, trip extra dims from beginning
        ndim = optical_control_image.ndim
        log.debug(f"image has shape {ndim}")
        while optical_control_image.ndim > 4:
            optical_control_image = optical_control_image[0, ...]
            ndim = optical_control_image.ndim
            log.debug(f"image has shape {ndim}")

        # detect center z-slice on reference channel
        log.debug("detecing center z in ref")
        ref_center_z = get_center_z(
            img_stack=optical_control_image[reference_channel, :, :, :]
        )

        # Crop with all available rings
        log.debug("crop rings")
        ref_crop, crop_dims = CropRings(
            img=optical_control_image[reference_channel, ref_center_z, :, :],
            pixel_size=px_size_xy,
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
        tform, align_info = RingAlignment(
            ref_seg_rings,
            ref_seg_rings_label,
            ref_props_df,
            ref_cross_label,
            mov_seg_rings,
            mov_seg_rings_label,
            mov_props_df,
            mov_cross_label,
        ).run()

        return tform, align_info


    def similarity_matrix_transform(
        self,
        alignment_matrix: numpy.ndarray,
        slice: numpy.ndarray,
    ) -> numpy.ndarray:
        if len(slice.shape) == 2:
            after_transform = transform.warp(
                slice, inverse_map=alignment_matrix, order=3
            )
        elif len(slice.shape) == 3:
            after_transform = numpy.zeros(slice.shape)
            for z in range(0, after_transform.shape[0]):
                after_transform[z, :, :] = transform.warp(
                    slice[z, :, :], inverse_map=alignment_matrix, order=3
                )
        else:
            raise IncompatibleImageException(
                f"Cannot perform similarity matrix transform: invalid image dimensions. \
                Image must be 2D or 3D but detected {len(slice.shape)} dimensions"
            )

        return (after_transform * 65535).astype(numpy.uint16)

    def align_image(
        self,
        alignment_matrix: numpy.ndarray,
        image: numpy.ndarray,
        channels_to_align: Dict[str, int],
    ) -> numpy.ndarray:

        aligned_image = numpy.zeros(image.shape)
        for channel, index in channels_to_align.items():
            if channel in ("Raw brightfield", "Raw 638nm"):
                aligned_slice = self.similarity_matrix_transform(
                    alignment_matrix, image[index]
                )
                aligned_image[index] = aligned_slice
            elif channel in ("Raw 488nm", "Raw 405nm", "Raw 561nm"):
                aligned_image[index] = image[index]

        return aligned_image.astype(numpy.uint16)

    def get_channel_name_to_index_map(self, image: AICSImage) -> Dict[str, int]:
        """
        Maps all channels in a split image to an index of where the channel is in the image
        Format is {"Raw channel_name" : Channel index}
        """
        channels = image.channel_names
        channel_info_dict = dict()
        for channel in channels:
            if channel in ["Bright_2", "TL_100x"]:
                channel_info_dict.update({"Raw brightfield": channels.index(channel)})
            elif channel in ["EGFP"]:
                channel_info_dict.update({"Raw 488nm": channels.index(channel)})
            elif channel in ["CMDRP"]:
                channel_info_dict.update({"Raw 638nm": channels.index(channel)})
            elif channel in ["H3342"]:
                channel_info_dict.update({"Raw 405nm": channels.index(channel)})
            elif channel in ["TaRFP"]:
                channel_info_dict.update({"Raw 561nm": channels.index(channel)})

        return channel_info_dict

    def crop(self):
        raise NotImplementedError("align_image")
