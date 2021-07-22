import logging
import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from skimage import measure

from camera_alignment_core.constants import (
    LOGGER_NAME,
)

from .segment_argolight_rings import SegmentRings


class CropRings(object):
    def __init__(
        self,
        img: NDArray[np.uint16],
        pixel_size: float,
        magnification: int,
        filter_px_size: int = 50,
    ):

        self.img = img
        # self.bead_dist_px = 15 / (pixel_size / 10 ** -6)
        self.bead_dist_px = 15 / pixel_size
        self.filter_px_size = filter_px_size
        self.magnification = magnification
        self.log = logging.getLogger(LOGGER_NAME)

        self.show_seg = False

    def get_crop_dimensions(
        self,
        img: NDArray[np.uint16],
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
        if cross_y % bead_dist_px > (bead_dist_px * crop_param):
            crop_top = 0
        else:
            crop_top = round(
                cross_y
                - (math.floor(cross_y / bead_dist_px) - (1 - crop_param)) * bead_dist_px
            )

        if (img.shape[0] - cross_y) % bead_dist_px > (bead_dist_px * crop_param):
            crop_bottom = img.shape[0]
        else:
            crop_bottom = img.shape[0] - round(
                img.shape[0]
                - (
                    cross_y
                    + (
                        math.floor((img.shape[0] - cross_y) / bead_dist_px)
                        - (1 - crop_param)
                    )
                    * bead_dist_px
                )
            )

        if cross_x % bead_dist_px > (bead_dist_px * crop_param):
            crop_left = 0
        else:
            crop_left = round(
                cross_x
                - (math.floor(cross_x / bead_dist_px) - (1 - crop_param)) * bead_dist_px
            )

        if (img.shape[1] - cross_x) % bead_dist_px > (bead_dist_px * crop_param):
            crop_right = img.shape[1]
        else:
            crop_right = img.shape[1] - round(
                img.shape[1]
                - (
                    cross_x
                    + (
                        math.floor((img.shape[1] - cross_x) / bead_dist_px)
                        - (1 - crop_param)
                    )
                    * bead_dist_px
                )
            )

        return crop_top, crop_bottom, crop_left, crop_right

    def make_grid(
        self, img: NDArray[np.uint16], cross_y: int, cross_x: int, bead_dist_px: float
    ) -> NDArray[np.bool_]:
        grid = np.zeros(img.shape)

        for y in np.arange(cross_y, 0, -bead_dist_px):
            for x in np.arange(cross_x, 0, -bead_dist_px):
                grid[int(y), int(x)] = True
            for x in np.arange(cross_x, img.shape[1], bead_dist_px):
                grid[int(y), int(x)] = True

        for y in np.arange(cross_y, img.shape[0], bead_dist_px):
            for x in np.arange(cross_x, 0, -bead_dist_px):
                grid[int(y), int(x)] = True
            for x in np.arange(cross_x, img.shape[1], bead_dist_px):
                grid[int(y), int(x)] = True

        return grid

    def run(
        self,
    ) -> tuple[
        NDArray[np.uint16],
        tuple[int, int, int, int],
        NDArray[np.bool_],
        pd.DataFrame,
        int,
        int,
    ]:
        self.log.debug("segment rings")

        seg_cross, props = SegmentRings(
            self.img, self.filter_px_size, self.magnification, thresh=None
        ).segment_cross(img=self.img, input_mult_factor=2.5)

        cross_y, cross_x = (
            props.loc[
                props["area"] == props["area"].max(), "centroid-0"
            ].values.tolist()[0],
            props.loc[
                props["area"] == props["area"].max(), "centroid-1"
            ].values.tolist()[0],
        )

        if self.magnification < 63:
            self.log.debug("get crop dimensions")
            crop_top, crop_bottom, crop_left, crop_right = self.get_crop_dimensions(
                self.img, int(cross_y), int(cross_x), self.bead_dist_px
            )
        else:
            crop_top = 0
            crop_left = 0
            crop_bottom = self.img.shape[0]
            crop_right = self.img.shape[1]

        crop_dimensions = (crop_top, crop_bottom, crop_left, crop_right)

        self.log.debug(f"crop dimensions {crop_dimensions}")
        img_out = self.img[crop_top:crop_bottom, crop_left:crop_right]

        updated_cross_y = cross_y - crop_bottom
        updated_cross_x = cross_x - crop_left

        self.log.debug(f"cross_y {updated_cross_y}")
        self.log.debug(f"cross_x {updated_cross_x}")
        self.log.debug("making grid")
        grid = self.make_grid(
            img_out, int(updated_cross_y), int(updated_cross_x), self.bead_dist_px
        )

        self.log.debug("label image")
        labelled_grid = measure.label(grid)
        props = measure.regionprops_table(
            labelled_grid, properties=["label", "area", "centroid"]
        )
        props_grid = pd.DataFrame(props)
        center_cross_label = labelled_grid[int(updated_cross_y), int(updated_cross_x)]

        number_of_rings = len(props)

        return (
            img_out,
            crop_dimensions,
            labelled_grid,
            props_grid,
            center_cross_label,
            number_of_rings,
        )
