import logging
from typing import List, Tuple

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
from .constants import LOGGER_NAME, Magnification
from .exception import (
    IncompatibleImageException,
    UnsupportedMagnification,
)

log = logging.getLogger(LOGGER_NAME)


def generate_alignment_matrix(
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

    if magnification not in [
        supported_magnification.value for supported_magnification in list(Magnification)
    ]:
        raise UnsupportedMagnification(
            f"Cannot perform image alignment for magnification {str(magnification)}."
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
        _ref_seg_rings,
        _ref_seg_rings_label,
        ref_props_df,
        ref_cross_label,
    ) = SegmentRings(ref_crop, px_size_xy, magnification, thresh=None).run()

    # segment rings on moving image
    log.debug("segment rings in moving")
    (
        _mov_seg_rings,
        _mov_seg_rings_label,
        mov_props_df,
        mov_cross_label,
    ) = SegmentRings(mov_crop, px_size_xy, magnification, thresh=None).run()

    # Create alignment from segmentation
    log.debug("Creating alignment matrix")
    similarity_transform, align_info = RingAlignment(
        ref_props_df,
        ref_cross_label,
        mov_props_df,
        mov_cross_label,
    ).run()

    return similarity_transform.params, align_info


def align_image(
    image: numpy.typing.NDArray[numpy.uint16],
    alignment_matrix: numpy.typing.NDArray[numpy.float16],
    channels_to_shift: List[int],
    interpolation: int = 0,
) -> numpy.typing.NDArray[numpy.uint16]:
    """Align a CZYX `image` using `alignment_matrix`.
    Will only apply the `alignment_matrix` to image slices within the channels specified in
    `channels_to_shift`.

    Parameters
    ----------
    image : numpy.typing.NDArray[numpy.uint16]
        Must be a 4 dimensional image in following dimensional order: 'CZYX'
    alignment_matrix : numpy.typing.NDArray[numpy.float16]
        3x3 matrix that can be used by skimage.transform.warp to transform a single z-slice of an image.
    channels_to_shift : List[int]
        Index positions of channels within `image` that should be shifted. N.b.: indices start at 0.
        E.g.: Specify [0, 2] to apply the alignment transform to channels at index positions 0 and 2 within `image`.
    interpolation : int
        Interpolation order to use when applying the alignment transform. Default is 0.
    """
    if not image.ndim == 4:
        raise IncompatibleImageException(
            f"Expected image to be 4 dimensional ('CZYX'). Got: {image.shape}"
        )

    if not channels_to_shift:
        raise ValueError(
            "channels_to_shift: passed an empty list to `align_image`. Cannot determine which channels to shift."
        )

    aligned_image = numpy.empty(image.shape, dtype=numpy.uint16)
    number_of_channels, *_ = image.shape
    for channel_index in range(0, number_of_channels):
        unaligned_channel = image[channel_index]
        if channel_index in channels_to_shift:
            log.debug("Applying alignment to %s channel", channel_index)
            aligned_channel = numpy.empty(unaligned_channel.shape, dtype=numpy.double)
            for z_index in range(0, aligned_channel.shape[0]):
                aligned_channel[z_index, :, :] = skimage.transform.warp(
                    unaligned_channel[z_index, :, :],
                    inverse_map=alignment_matrix,
                    order=interpolation,  # 3
                    preserve_range=True,
                )

            aligned_image[channel_index] = aligned_channel
        else:
            log.debug("Skipping alignment for %s channel", channel_index)
            aligned_image[channel_index] = unaligned_channel

    return aligned_image


def crop(
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
    log.debug("Cropping to <X: %s, Y: %s>", cropping_dimension.x, cropping_dimension.y)

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

    # Check if there are black pixels, if so, log a warning
    if numpy.any(cropped_image < black_pixel_cutoff):
        log.warning(
            "Black pixels are detected, either from original image or due to alignment."
        )

    return cropped_image
