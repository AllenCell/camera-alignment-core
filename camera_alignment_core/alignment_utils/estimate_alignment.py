from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from skimage import transform as tf
from skimage.measure import ransac

from .alignment_info import AlignmentInfo


class RingAlignment(object):
    def __init__(
        self,
        ref_seg_rings: NDArray[np.uint16],
        ref_label_rings: NDArray[np.uint16],
        ref_rings_props: pd.DataFrame,
        ref_cross_label: int,
        mov_seg_rings: NDArray[np.uint16],
        mov_label_rings: NDArray[np.uint16],
        mov_rings_props: pd.DataFrame,
        mov_cross_label: int,
        alignment_method: str,
    ):
        self.ref_seg_rings = ref_seg_rings
        self.mov_label_rings = ref_label_rings
        self.ref_rings_props = ref_rings_props
        self.ref_cross_label = ref_cross_label
        self.mov_seg_rings = mov_seg_rings
        self.mov_label_rings = mov_label_rings
        self.mov_rings_props = mov_rings_props
        self.mov_cross_label = mov_cross_label
        self.alignment_method = alignment_method

    def assign_ref_to_mov(
        self,
        updated_ref_peak_dict: Dict[int, Tuple[int, int]],
        updated_mov_peak_dict: Dict[int, Tuple[int, int]],
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[Tuple[int, int], Tuple[int, int]]]:
        """
        Assigns beads from moving image to reference image using
        linear_sum_assignment to reduce the distance between the same bead on
        the separate channels. In case where there is more beads in one channel
        than the other, this method will throw off the extra bead that cannot
        be assigned to one, single bead on the other image.

        :param updated_ref_peak_dict: A dictionary
            ({bead_number: (coor_y, coor_x)}) from reference beads
        :param updated_mov_peak_dict:  A dictionary
            ({bead_number: (coor_y, coor_x)}) from moving beads
        :return:
            bead_peak_intensity_dict: A full dictionary mapping the reference bead number and coordinates with the moving
                                      ones. ({(bead_ref_number, (bead_ref_coor_y, bead_ref_coor_x)): (bead_mov_number, (bead_mov_coor_y, bead_mov_coor_x))
            ref_mov_num_dict: A dictionary mapping the reference bead number and moving bead number
            ref_mov_coor_dict: A dictionary mapping the reference bead coordinates and moving bead coordinates
        """
        updated_ref_peak = list(OrderedDict(updated_ref_peak_dict).items())
        updated_mov_peak = list(OrderedDict(updated_mov_peak_dict).items())

        dist_tx = np.zeros((len(updated_ref_peak), len(updated_mov_peak)))
        for i, (bead_ref, coor_ref) in enumerate(updated_ref_peak):
            for j, (bead_mov, coor_mov) in enumerate(updated_mov_peak):
                dist_tx[i, j] = distance.euclidean(coor_ref, coor_mov)

        ref_ind, mov_ind = linear_sum_assignment(dist_tx)

        bead_peak_intensity_dict = {}
        ref_mov_num_dict = {}
        ref_mov_coor_dict = {}
        for num_bead in range(0, len(ref_ind)):
            bead_peak_intensity_dict.update(
                {
                    updated_ref_peak[ref_ind[num_bead]]: updated_mov_peak[
                        mov_ind[num_bead]
                    ]
                }
            )
            ref_mov_num_dict.update(
                {
                    updated_ref_peak[ref_ind[num_bead]][0]: updated_mov_peak[
                        mov_ind[num_bead]
                    ][0]
                }
            )
            ref_mov_coor_dict.update(
                {
                    updated_ref_peak[ref_ind[num_bead]][1]: updated_mov_peak[
                        mov_ind[num_bead]
                    ][1]
                }
            )

        return bead_peak_intensity_dict, ref_mov_num_dict, ref_mov_coor_dict

    def rings_coor_dict(
        self, props: pd.DataFrame, cross_label: int
    ) -> Dict[int, Tuple[int, int]]:
        """
        Generate a dictionary from regionprops_table in the form of {label: (coor_y, coor_x)} for rings image
        :param props: a dataframe containing regionprops_table output
        :param cross_label: Integer value representing where the center cross is in the rings image
        :return:
            img_dict: A dictionary of label to coordinates
        """
        img_dict = {}
        for index, row in props.iterrows():
            if row["label"] is not cross_label:
                img_dict.update({row["label"]: (row["centroid-0"], row["centroid-1"])})

        return img_dict

    def change_coor_system(
        self, coor_dict: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Changes coordinates in a dictionary from {(y1, x1):(y2, x2)} to {(x1, y1): (x2, y2)}
        :param coor_dict: A dictionary of coordinates in the form of {(y1, x1):(y2, x2)}
        :return:
            An updated reversed coordinate dictionary that is {(x1, y1): (x2, y2)}
        """
        rev_yx_to_xy = {}
        for coor_ref, coor_mov in coor_dict.items():
            rev_yx_to_xy.update(
                {(coor_ref[1], coor_ref[0]): (coor_mov[1], coor_mov[0])}
            )
        return rev_yx_to_xy

    def report_number_beads(
        self, bead_dict: Dict[int, int], method_logging=True
    ) -> Tuple[bool, int]:
        """
        Reports the number of beads used to estimate transform
        :param bead_dict: A dictionary that each key is a bead
        :param method_logging: A boolean to indicate if user wants print statements
        :return:
            bead_num_qc: Boolean indicates if number of beads passed QC (>=10) or failed (<10)
            num_beads: An integer of number of beads used
        """
        bead_num_qc = False
        num_beads = len(bead_dict)
        if num_beads >= 10:
            bead_num_qc = True
        if method_logging:
            print("number of beads used to estimate transform: " + str(num_beads))
        return bead_num_qc, num_beads

    def report_similarity_matrix_parameters(
        self, tform: tf.SimilarityTransform, method_logging=True
    ) -> AlignmentInfo:
        """
        Reports similarity matrix and its parameters
        :param tform: A transform generated from skimage.transform.estimate_transform
        :param method_logging: A boolean to indicate if printing/logging statements is selected
        :return: A dictionary with the following keys and values:
            transform: A transform to be applied to moving images
            scaling: Uniform scaling parameter
            shift_y: Shift in y
            shift_x: Shift in x
            rotate_angle: Rotation angle
        """
        tform_dict = {
            "transform": tform,
            "scaling": tform.scale,
            "shift_y": tform.translation[0],
            "shift_x": tform.translation[1],
            "rotate_angle": tform.rotation,
        }

        if method_logging:
            for param, value in tform_dict.items():
                print(param + ": " + str(value))

        align_info = AlignmentInfo(
            tform=tform,
            rotation=tform.rotation,
            shift_x=tform.translation[1],
            shift_y=tform.translation[0],
            z_offset=0,
            scaling=tform.scale,
        )

        return align_info

    def run(
        self,
    ) -> Tuple[
        tf.SimilarityTransform,
        Dict[Tuple[int, int], Tuple[int, int]],
        AlignmentInfo,
        Tuple[bool, int],
    ]:
        ref_centroid_dict = self.rings_coor_dict(
            self.ref_rings_props, self.ref_cross_label
        )
        mov_centroid_dict = self.rings_coor_dict(
            self.mov_rings_props, self.mov_cross_label
        )

        (
            bead_centroid_dict,
            ref_mov_num_dict,
            ref_mov_coor_dict,
        ) = self.assign_ref_to_mov(ref_centroid_dict, mov_centroid_dict)

        rev_coor_dict = self.change_coor_system(ref_mov_coor_dict)
        print(rev_coor_dict)

        if self.alignment_method == "alignV2":
            # alignV2 method
            tform = tf.estimate_transform(
                "similarity",
                np.asarray(list(rev_coor_dict.keys())),
                np.asarray(list(rev_coor_dict.values())),
            )

        elif self.alignment_method == "ransac":
            # RnD method with ransac
            # robustly estimate affine transform model with RANSAC

            tform, inliers = ransac(
                (
                    np.asarray(list(rev_coor_dict.keys())),
                    np.asarray(list(rev_coor_dict.values())),
                ),
                tf.SimilarityTransform,
                min_samples=3,
                residual_threshold=2,
                max_trials=100,
            )

        align_info = self.report_similarity_matrix_parameters(tform)
        num_beads_for_estimation = self.report_number_beads(bead_centroid_dict)

        return tform, rev_coor_dict, align_info, num_beads_for_estimation
