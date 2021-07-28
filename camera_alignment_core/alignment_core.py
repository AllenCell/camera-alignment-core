import logging
from typing import Dict, List, Tuple

from aicsimageio import AICSImage
import numpy
import numpy.typing

from .alignment_utils.alignment_info import (
    AlignmentInfo,
)
from .alignment_utils.crop_argolight_rings_img import (
    CropRings,
)
from .alignment_utils.estimate_alignment import (
    RingAlignment,
)
from .alignment_utils.get_center_z import (
    GetCenterZ,
)
from .alignment_utils.segment_argolight_rings import (
    SegmentRings,
)
from .constants import LOGGER_NAME

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
        ref_center_z, _ = GetCenterZ(
            img_stack=optical_control_image[reference_channel, :, :, :]
        ).run()

        # Crop with all available rings
        log.debug("crop rings")
        ref_crop, crop_dims, _, _, _, _ = CropRings(
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
        ) = SegmentRings(
            ref_crop, px_size_xy, magnification, thresh=None, debug_mode=False
        ).run()

        # segment rings on moving image
        log.debug("segment rings in moving")
        (
            mov_seg_rings,
            mov_seg_rings_label,
            mov_props_df,
            mov_cross_label,
        ) = SegmentRings(
            mov_crop, px_size_xy, magnification, thresh=None, debug_mode=False
        ).run()

        # estimate alignment from segmentation
        log.debug("estimating alignment matrix")
        tform, _, align_info, _ = RingAlignment(
            ref_seg_rings,
            ref_seg_rings_label,
            ref_props_df,
            ref_cross_label,
            mov_seg_rings,
            mov_seg_rings_label,
            mov_props_df,
            mov_cross_label,
            "alignV2",
        ).run()

        return tform, align_info

    def align_image(
        self,
        alignment_matrix: numpy.typing.NDArray[numpy.float16],
        image: numpy.typing.NDArray[numpy.uint16],
        channels_to_align: List[int],
    ) -> numpy.typing.NDArray[numpy.uint16]:
        raise NotImplementedError("align_image")

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
