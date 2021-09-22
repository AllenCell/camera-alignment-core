import logging
import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..constants import LOGGER_NAME
from .segment_rings import SegmentRings

log = logging.getLogger(LOGGER_NAME)

MIN_NO_CROP_MAGNIFICATION = 63
SEGMENTATION_MULT_FACTOR = 2.5
BEAD_DISTANCE_UM = 15


class CropRings:
    def __init__(
        self,
        img: NDArray[np.uint16],
        pixel_size_um: float,
        magnification: int,
        filter_px_size: int = 50,
        bead_distance: int = BEAD_DISTANCE_UM,
    ):
        self.img = img
        self.bead_dist_px = bead_distance / pixel_size_um
        self.filter_px_size = filter_px_size
        self.magnification = magnification

    @staticmethod
    def get_crop_dimensions(
        img_height: int,
        img_width: int,
        cross_y: int,
        cross_x: int,
        bead_dist_px: float,
        crop_param: float = 0.5,
    ) -> Tuple[int, int, int, int]:
        """
        Calculates the crop dimension from the location of the cross to capture complete rings in the image
        Parameters
        ----------
        img: mxn nd-array image of rings
        cross_y: y location of center cross
        cross_x: x location of center cross
        bead_dist_px: distance between rings in pixels
        crop_param: a float between 0 and 1 that indicates a factor of distance between rings that should be left behind
            after cropping

        Returns
        -------
        crop_top: top pixels to keep
        crop_bottom: bottom pixels to keep
        crop_left: left pixels to keep
        crop_right: right pixels to keep
        """
        # get crop for top of image
        if cross_y % bead_dist_px > (bead_dist_px * crop_param):
            crop_top = 0
        else:
            crop_top = round(
                cross_y
                - (math.floor(cross_y / bead_dist_px) - (1 - crop_param)) * bead_dist_px
            )

        # get crop for bottom of image
        if (img_height - cross_y) % bead_dist_px > (bead_dist_px * crop_param):
            crop_bottom = img_height
        else:
            crop_bottom = img_height - round(
                img_height
                - (
                    cross_y
                    + (
                        math.floor((img_height - cross_y) / bead_dist_px)
                        - (1 - crop_param)
                    )
                    * bead_dist_px
                )
            )

        # get crop for left side of image
        if cross_x % bead_dist_px > (bead_dist_px * crop_param):
            crop_left = 0
        else:
            crop_left = round(
                cross_x
                - (math.floor(cross_x / bead_dist_px) - (1 - crop_param)) * bead_dist_px
            )

        # get crop for right side of image
        if (img_width - cross_x) % bead_dist_px > (bead_dist_px * crop_param):
            crop_right = img_width
        else:
            crop_right = img_width - round(
                img_width
                - (
                    cross_x
                    + (
                        math.floor((img_width - cross_x) / bead_dist_px)
                        - (1 - crop_param)
                    )
                    * bead_dist_px
                )
            )

        return crop_top, crop_bottom, crop_left, crop_right

    def run(
        self,
        min_no_crop_magnification: int = MIN_NO_CROP_MAGNIFICATION,
        segmentation_mult_factor: float = SEGMENTATION_MULT_FACTOR,
    ) -> Tuple[NDArray[np.uint16], Tuple[int, int, int, int]]:
        """
        Crop image closer to ring grid
        :min_no_crop_magnification: int
            Minimum magnification at which we do not need to crop the bead image (e.g., because it's zoomed enough)
        :segmentation_mult_factor: float
            Value passed directly to SegmentRings::segment_cross as `input_mult_factor`
        """
        log.debug("segment rings")
        # segment rings in image
        _, props = SegmentRings(
            self.img, self.filter_px_size, self.magnification, thresh=None
        ).segment_cross(img=self.img, input_mult_factor=segmentation_mult_factor)

        # find centroid of cross (assumed to be largest object)
        cross_y, cross_x = (
            props.loc[
                props["area"] == props["area"].max(), "centroid-0"
            ].values.tolist()[0],
            props.loc[
                props["area"] == props["area"].max(), "centroid-1"
            ].values.tolist()[0],
        )

        # crop if image is below minimum maginification
        if self.magnification < min_no_crop_magnification:
            log.debug("Determining cropping dimensions")
            crop_top, crop_bottom, crop_left, crop_right = self.get_crop_dimensions(
                self.img.shape[0],
                self.img.shape[1],
                int(cross_y),
                int(cross_x),
                self.bead_dist_px,
            )
        else:
            crop_top = 0
            crop_left = 0
            crop_bottom = self.img.shape[0]
            crop_right = self.img.shape[1]

        crop_dimensions = (crop_top, crop_bottom, crop_left, crop_right)

        # crop image
        log.debug(
            "Crop dimensions: <Top: %s, Bottom: %s, Left: %s: Right: %s>",
            crop_top,
            crop_bottom,
            crop_left,
            crop_right,
        )
        img_out = self.img[crop_top:crop_bottom, crop_left:crop_right]

        return (
            img_out,
            crop_dimensions,
        )
