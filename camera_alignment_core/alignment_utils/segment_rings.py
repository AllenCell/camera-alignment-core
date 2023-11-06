import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure as exp
from skimage import filters, measure, morphology
from skimage.morphology import (
    remove_small_objects,
)

from ..constants import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)

BEAD_DISTANCE_UM = 15
CROSS_SIZE_UM = 7.5 * 6 * 10**-6
RING_RADIUS_UM = 0.7 * 10**-6


class SegmentRings:
    def __init__(
        self,
        img: np.typing.NDArray[np.uint16],
        pixel_size: float,
        magnification: int,
        thresh: Optional[Tuple[float, float]] = None,
        bead_distance_um: float = BEAD_DISTANCE_UM,
        cross_size_um: float = CROSS_SIZE_UM,
        ring_radius_um: float = RING_RADIUS_UM,
    ):
        self.img = img
        self.pixel_size = pixel_size
        self.magnification = magnification

        self.cross_size_px = cross_size_um / self.pixel_size
        self.ring_size_px = math.pi * (ring_radius_um / self.pixel_size) ** 2
        self.bead_dist_px = bead_distance_um / self.pixel_size

        if thresh is not None:
            self.thresh = thresh
        elif self.magnification in [40, 63, 100]:
            self.thresh = (0.2, 99.8)
        else:
            self.thresh = (0.5, 99.5)

    def dot_2d_slice_by_slice_wrapper(
        self, struct_img: np.typing.NDArray[np.float32], s2_param: List
    ) -> np.typing.NDArray[np.bool_]:
        """
        https://github.com/AllenCell/aics-segmentation/blob/main/aicssegmentation/core/seg_dot.py
        wrapper for 2D spot filter on 3D image slice by slice
        Parameters:
        ------------
        struct_img: np.typing.NDArray
            a 3d numpy array, usually the image after smoothing
        s2_param: List
            [[scale_1, cutoff_1], [scale_2, cutoff_2], ....], e.g. [[1, 0.1]]
            or [[1, 0.12], [3,0.1]]: scale_x is set based on the estimated radius
            of your target dots. For example, if visually the diameter of the
            dots is usually 3~4 pixels, then you may want to set scale_x as 1
            or something near 1 (like 1.25). Multiple scales can be used, if
            you have dots of very different sizes. cutoff_x is a threshold
            applied on the actual filter reponse to get the binary result.
            Smaller cutoff_x may yielf more dots and fatter segmentation,
            while larger cutoff_x could be less permisive and yield less
            dots and slimmer segmentation.
        """
        bw = np.zeros(struct_img.shape, dtype=bool)
        for fid in range(len(s2_param)):
            log_sigma = s2_param[fid][0]
            responce = np.zeros_like(struct_img)
            for zz in range(struct_img.shape[0]):
                responce[zz, :, :] = (
                    -1
                    * (log_sigma**2)
                    * ndi.filters.gaussian_laplace(struct_img[zz, :, :], log_sigma)
                )
            bw = np.logical_or(bw, responce > s2_param[fid][1])
        return bw

    def preprocess_img(self) -> np.typing.NDArray[np.uint16]:
        """
        Pre-process image with raw-intensity with rescaling and smoothing using pre-defined parameters from image
        magnification information
        Returns
        -------
        smooth: smooth image
        """
        rescale = exp.rescale_intensity(
            self.img,
            in_range=(
                np.percentile(self.img, self.thresh[0]),
                np.percentile(self.img, self.thresh[1]),
            ),
        )
        smooth = filters.gaussian(rescale, sigma=1, preserve_range=False)
        return smooth

    def segment_cross(
        self,
        img: np.typing.NDArray[np.uint16],
        mult_factor_range: Tuple[int, int] = (1, 5),
        input_mult_factor: float = None,
    ) -> Tuple[np.typing.NDArray[np.bool_], pd.DataFrame]:
        """
        Segments the center cross in the image through iterating the intensity-threshold parameter until one object
        greater than the expected cross size (in pixel) is segmented
        Parameters
        ----------
        img: image (intensity after smoothing)
        mult_factor_range: range of multiplication factor to determine threshold for segmentation

        Returns
        -------
        seg_cross: binary image of segmented cross
        props: dataframe describing the centroid location and size of the segmented cross
        """
        if input_mult_factor is not None:
            _, label_for_cross = self.segment_rings_intensity_threshold(
                img, mult_factor=input_mult_factor
            )
        else:
            for mult_factor in np.linspace(mult_factor_range[1], mult_factor_range[0]):
                _, label_for_cross = self.segment_rings_intensity_threshold(
                    img, mult_factor=mult_factor
                )
                if (
                    np.max(label_for_cross) >= 1
                    and np.sum(label_for_cross) > self.cross_size_px
                ):
                    break

        # try:
        _, props, cross_label = self.filter_center_cross(label_for_cross)
        # except:
        #     from skimage.io import imsave
        #     import pdb; pdb.set_trace()

        seg_cross = label_for_cross == cross_label

        return seg_cross, props

    def segment_rings_intensity_threshold(
        self,
        img: np.typing.NDArray[np.uint16],
        filter_px_size=50,
        mult_factor=2.5,
    ) -> Tuple[np.typing.NDArray[np.bool_], np.typing.NDArray[np.uint16]]:
        """
        Segments rings using intensity-thresholded method
        Parameters
        ----------
        img: rings image (after smoothing)
        filter_px_size: any segmented below this size will be filtered out
        mult_factor: parameter to adjust threshold
        show_seg: boolean to display segmentation

        Returns
        -------
        filtered_seg: binary mask of ring segmentation
        filtered_label: labelled mask of ring segmentation
        """
        thresh = np.median(img) + mult_factor * np.std(img)
        seg = np.zeros(img.shape, dtype=np.bool_)
        seg[img >= thresh] = True

        seg = remove_small_objects(seg > 0, int(filter_px_size))
        labelled_seg = measure.label(seg).astype(np.uint16)

        return seg, labelled_seg

    def segment_rings_dot_filter(
        self,
        img_2d: np.typing.NDArray[np.uint16],
        seg_cross: np.typing.NDArray[np.bool_],
        num_beads: int,
        minArea: int,
        search_range: Tuple[float, float] = (0, 0.75),
        size_param: float = 2.5,
    ) -> Tuple[
        np.typing.NDArray[np.bool_], np.typing.NDArray[np.uint16], Optional[float]
    ]:
        """
        Segments rings using 2D dot filter from aics-segmenter. The method loops through a possible range of parameters
        and automatically detects the optimal filter parameter when it segments the number of expected rings objects
        Parameters
        ----------
        img: rings image (after smoothing)
        seg_cross: binary mask of the center cross object
        num_beads: expected number of beads
        minArea: minimum area of rings, any segmented object below this size will be filtered out
        search_range: initial search range of filter parameter
        size_param: size parameter of dot filter

        Returns
        -------
        seg: binary mask of ring segmentation
        label: labelled mask of ring segmentation
        thresh: filter parameter after optimization

        """
        img = np.zeros((1, img_2d.shape[0], img_2d.shape[1]), dtype=np.float32)
        img[0, :, :] = img_2d

        thresh = None
        for seg_param in np.linspace(search_range[1], search_range[0], 500):
            s2_param = [[size_param, seg_param]]
            seg = self.dot_2d_slice_by_slice_wrapper(img, s2_param)[0, :, :]

            remove_small = remove_small_objects(
                seg > 0, min_size=minArea, connectivity=1, in_place=False
            )

            dilate = morphology.binary_dilation(remove_small, selem=morphology.disk(2))
            seg_rings = morphology.binary_erosion(dilate, selem=morphology.disk(2))

            seg = np.logical_or(seg_cross, seg_rings).astype(np.bool_)
            label = measure.label(seg).astype(np.uint16)

            if np.max(label) >= num_beads:
                thresh = float(seg_param)
                break

        return seg, label, thresh

    def filter_center_cross(
        self, label_seg: np.typing.NDArray[np.uint16]
    ) -> Tuple[np.typing.NDArray[np.uint16], pd.DataFrame, int]:
        """
        filters out where the center cross (the biggest segmented object) is in a labelled rings image

        Parameters
        ----------
        label_seg: A labelled image

        Returns
        -------
        filter_label: A labelled image after filtering the center cross (center cross = 0)
        props_df: A dataframe from regionprops_table with columns ['label', 'centroid-0', 'centroid-y', 'area']
        cross_label: The integer label of center cross

        """

        props = measure.regionprops_table(
            label_seg, properties=["label", "area", "centroid"]
        )
        props_df = pd.DataFrame(props)

        cross_label = props_df.loc[
            (props_df["area"] == props_df["area"].max()), "label"
        ].values.tolist()[0]

        filter_label = label_seg.copy()
        filter_label[label_seg == cross_label] = 0

        return filter_label, props_df, cross_label

    def get_number_rings(
        self, img: np.typing.NDArray[np.uint16], mult_factor: int = 5
    ) -> int:
        """
        Estimates the number of rings in a rings object using the location of the center cross
        Parameters
        ----------
        img: input image (after smoothing)
        bead_dist_px: calculated distance between rings in pixels
        mult_factor: parameter to segment cross with

        Returns
        -------
        num_beads: number of beads after estimation
        """
        # update cross info
        _, props = self.segment_cross(img, input_mult_factor=mult_factor)

        # get number of beads from the location of center of cross
        cross_y, cross_x = (
            props.loc[
                props["area"] == props["area"].max(), "centroid-0"
            ].values.tolist()[0],
            props.loc[
                props["area"] == props["area"].max(), "centroid-1"
            ].values.tolist()[0],
        )

        num_beads = (
            math.floor(cross_y / self.bead_dist_px)
            + math.floor((img.shape[0] - cross_y) / self.bead_dist_px)
            + 1
        ) * (
            math.floor(cross_x / self.bead_dist_px)
            + math.floor((img.shape[1] - cross_x) / self.bead_dist_px)
            + 1
        )

        return num_beads

    def run(
        self,
    ) -> Tuple[
        np.typing.NDArray[np.bool_], np.typing.NDArray[np.uint16], pd.DataFrame, int
    ]:
        img_preprocessed = self.preprocess_img()

        num_beads = self.get_number_rings(img=img_preprocessed, mult_factor=5)
        minArea = int(self.ring_size_px * 0.8)

        if self.magnification in [40, 63, 100]:
            seg_rings, label_rings = self.segment_rings_intensity_threshold(
                img_preprocessed
            )
        else:
            seg_cross, _ = self.segment_cross(img=img_preprocessed)

            seg_rings, label_rings, _ = self.segment_rings_dot_filter(
                img_2d=img_preprocessed,
                seg_cross=seg_cross,
                num_beads=num_beads,
                minArea=minArea,
            )

        # try:
        _, props_df, cross_label = self.filter_center_cross(label_rings)
        # except:
        #     from skimage.io import imsave
        #     import pdb; pdb.set_trace()

        return seg_rings, label_rings, props_df, cross_label
