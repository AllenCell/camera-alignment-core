import logging
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance
from skimage import metrics
from skimage import transform as tf

from ..constants import LOGGER_NAME
from .get_center_z import get_center_z

log = logging.getLogger(LOGGER_NAME)


class AlignmentQC:
    def __init__(
        self,
        reference: NDArray[np.uint16] = None,
        moving: NDArray[np.uint16] = None,
        reference_seg: NDArray[np.uint16] = None,
        moving_seg: NDArray[np.uint16] = None,
        ref_mov_coor_dict: Dict[Tuple[int, int], Tuple[int, int]] = None,
        rev_coor_dict: Dict[Tuple[int, int], Tuple[int, int]] = None,
        tform: tf.SimilarityTransform = None,
    ):
        self.reference = reference
        self.moving_source = moving

        if reference is not None:
            self.ref_origin = get_center_z(reference)
        else:
            self.ref_origin = 0
        if moving is not None:
            self.mov_origin = get_center_z(moving)
        else:
            self.mov_origin = 0

        self.reference_seg = reference_seg
        self.moving_seg = moving_seg

        self.ref_mov_coor_dict = ref_mov_coor_dict
        self.rev_coor_dict = rev_coor_dict

        if moving is not None and tform is not None:
            self.moving_transformed = tf.warp(moving, inverse_map=tform, order=3)
        else:
            self.moving_transformed = None
        if self.moving_seg is not None and tform is not None:
            self.moving_seg_transformed = tf.warp(
                moving_seg, inverse_map=tform, order=0, preserve_range=True
            ).astype(np.uint16)
        else:
            self.moving_seg_transformed = None
        self.tform = tform

    # Full QC ################################################################

    def report_full_metrics(
        self,
    ) -> Dict[str, Optional[object]]:

        missing = self.check_all_defined()
        if missing is not None:
            log.error(
                "Error: the following variables have not been set - "
                + ",".join(missing)
            )

        z_offset, ref_origin, mov_origin = self.check_z_offset_between_ref_mov()
        ref_signal, ref_noise, mov_signal, mov_noise = self.report_ref_mov_image_snr()
        bead_num_qc, num_beads = self.report_number_beads()
        change_fov_intensity_param_dict = self.report_change_fov_intensity_parameters()
        coor_dist_qc, diff_sum_beads = self.report_changes_in_coordinates_mapping()
        mse_qc, diff_mse = self.report_changes_in_mse()

        qc_metrics = {
            "z_offset": z_offset,
            "reference_mid_z": ref_origin,
            "moving_mid_z": mov_origin,
            "reference_signal": ref_signal,
            "reference_noise": ref_noise,
            "moving_signal": mov_signal,
            "moving_noise": mov_noise,
            "bead_qc": bead_num_qc,
            "num_beads": num_beads,
            "change_fov_intensity_param_dict": change_fov_intensity_param_dict,
            "coor_dist_qc": coor_dist_qc,
            "diff_sum_beads": diff_sum_beads,
            "mse_qc": mse_qc,
            "diff_mse": diff_mse,
        }

        return qc_metrics

    # Set individual variables ################################################

    def set_raw_images(
        self,
        reference: NDArray[np.uint16],
        moving: NDArray[np.uint16],
    ):
        self.reference = reference
        self.moving_source = moving

        self.ref_origin = get_center_z(self.reference)
        self.mov_origin = get_center_z(self.moving_source)

        return

    def set_seg_images(
        self,
        reference_seg: NDArray[np.uint16],
        moving_seg: NDArray[np.uint16],
    ):
        self.reference_seg = reference_seg
        self.moving_seg = moving_seg
        return

    def set_ref_mov_coor_dict(
        self,
        ref_mov_coor_dict: Dict[Tuple[int, int], Tuple[int, int]],
    ):
        self.ref_mov_coor_dict = ref_mov_coor_dict
        return

    def set_rev_coor_dict(
        self,
        rev_coor_dict: Dict[Tuple[int, int], Tuple[int, int]],
    ):
        self.rev_coor_dict = rev_coor_dict
        return

    def set_tform(
        self,
        tform: tf.SimilarityTransform,
    ):
        self.tform = tform
        if self.moving_source is not None:
            log.info("Applying transform to current source image")
            self.moving_transformed = tf.warp(
                self.moving_source[self.mov_origin], inverse_map=tform, order=3
            )
        else:
            log.warning(
                "Moving source is not yet defined. Please define and run apply_transform() before qc"
            )
        if self.moving_seg is not None:
            log.info("Applying transform to current source image")
            self.moving_seg_transformed = tf.warp(
                self.moving_seg, inverse_map=tform, order=0, preserve_range=True
            )
        else:
            log.warning(
                "Moving seg is not yet defined. Please define and run apply_transform() before qc"
            )

    def apply_transform(self):
        if self.tform is not None:
            self.moving_transformed = tf.warp(
                self.moving_source[self.mov_origin], inverse_map=self.tform, order=3
            )
            self.moving_seg_transformed = tf.warp(
                self.moving_seg, inverse_map=self.tform, order=0, preserve_range=True
            )
        else:
            log.error("Error: tform is not yet defined")

    def check_all_defined(self) -> Optional[list[str]]:
        missing = []

        if self.reference is None:
            missing.append("source image")
        if self.moving_source is None:
            missing.append("moving source")
        if self.reference_seg is None:
            missing.append("reference seg")
        if self.moving_seg is None:
            missing.append("moving seg")
        if self.ref_mov_coor_dict is None:
            missing.append("reference moving coordinate dict")
        if self.rev_coor_dict is None:
            missing.append("ref coordinate dict")
        if self.moving_transformed is None:
            missing.append("moving transformed")
        if self.moving_seg_transformed is None:
            missing.append("moving seg transformed")
        if self.tform is None:
            missing.append("transformation")

        if len(missing) == 0:
            return None
        else:
            return missing

    # QC Functions ############################################################

    def check_z_offset_between_ref_mov(self) -> Tuple[int, int, int]:
        if self.reference is None or self.moving_source is None:
            log.error("Error: Raw images are missing for qc")
            raise Exception("Error: Raw images are missing for qc")

        z_offset = self.ref_origin - self.mov_origin

        return z_offset, self.ref_origin, self.mov_origin

    def report_ref_mov_image_snr(self) -> Tuple[int, int, int, int]:
        if self.reference is None or self.moving_source is None:
            log.error("Error: Seg images are missing for qc")
            raise Exception("Error: Seg images are missing for qc")

        def get_image_snr(
            seg: Optional[NDArray[np.uint16]],
            img_intensity: Optional[NDArray[np.uint16]],
        ) -> Tuple[int, int]:
            if seg is None or img_intensity is None:
                return -1, -1
            signal = np.median(img_intensity[seg.astype(bool)])
            noise = np.median(img_intensity[~seg.astype(bool)])

            return signal, noise

        ref_signal, ref_noise = get_image_snr(
            self.reference_seg, self.reference[self.ref_origin]
        )
        mov_signal, mov_noise = get_image_snr(
            self.moving_seg, self.moving_source[self.mov_origin]
        )

        return ref_signal, ref_noise, mov_signal, mov_noise

    def report_number_beads(self) -> Tuple[bool, int]:
        if self.ref_mov_coor_dict is None:
            log.error("Error: ref_mov_coor_dict is missing for qc")
            raise Exception("Error: ref_mov_coor_dict is missing for qc")

        bead_num_qc = False
        num_beads = len(self.ref_mov_coor_dict)
        if num_beads >= 10:
            bead_num_qc = True
        return bead_num_qc, num_beads

    def report_change_fov_intensity_parameters(self) -> Dict[str, int]:
        """
        Reports changes in FOV intensity after transform
        :return: A dictionary with the following keys and values:
            median_intensity
            min_intensity
            max_intensity
            1_percentile: first percentile intensity
            995th_percentile: 99.5th percentile intensity
        """
        if self.moving_transformed is None or self.moving_source is None:
            log.error("Error: moving source and transformed images are missing for qc")
            raise Exception(
                "Error: moving source and transformed images are missing for qc"
            )

        change_fov_intensity_param_dict = {
            "median_intensity": np.median(self.moving_transformed) * 65535
            - np.median(self.moving_source[self.mov_origin]),
            "min_intensity": np.min(self.moving_transformed) * 65535
            - np.min(self.moving_source[self.mov_origin]),
            "max_intensity": np.max(self.moving_transformed) * 65535
            - np.max(self.moving_source[self.mov_origin]),
            "1st_percentile": np.percentile(self.moving_transformed, 1) * 65535
            - np.percentile(self.moving_source[self.mov_origin], 1),
            "995th_percentile": np.percentile(self.moving_transformed, 99.5) * 65535
            - np.percentile(self.moving_source[self.mov_origin], 99.5),
        }

        return change_fov_intensity_param_dict

    def report_changes_in_coordinates_mapping(self) -> Tuple[bool, float]:
        """
        Report changes in beads (center of FOV) centroid coordinates before and after transform. A good transform will
        reduce the difference in distances, or at least not increase too much (thresh=5), between transformed_mov_beads and
        ref_beads than mov_beads and ref_beads. A bad transform will increase the difference in distances between
        transformed_mov_beads and ref_beads.
        """

        if self.ref_mov_coor_dict is None:
            log.error("Error: ref_mov_coor_dict is missing for qc")
            raise Exception("Error: ref_mov_coor_dict is missing for qc")

        if self.tform is None:
            log.error("Error: tform is missing for qc")
            raise Exception("Error: tform is missing for qc")

        if self.reference is None:
            log.error("Error: reference raw image is missing for qc")
            raise Exception("Error: reference raw image is missing for qc")

        transform_qc = False
        mov_coors = list(self.ref_mov_coor_dict.values())
        ref_coors = list(self.ref_mov_coor_dict.keys())
        mov_transformed_coors = self.tform(mov_coors)

        dist_before_list = []
        dist_after_list = []
        for bead in range(0, len(mov_coors)):
            dist_before = distance.euclidean(mov_coors[bead], ref_coors[bead])
            dist_after = distance.euclidean(
                mov_transformed_coors[bead], ref_coors[bead]
            )
            dist_before_list.append(dist_before)
            dist_after_list.append(dist_after)

        # filter center beads only
        y_size = 360
        x_size = 536

        y_lim = (
            int(self.reference[self.ref_origin].shape[0] / 2 - y_size / 2),
            int(self.reference[self.ref_origin].shape[0] / 2 + y_size / 2),
        )
        x_lim = (
            int(self.reference[self.ref_origin].shape[1] / 2 - x_size / 2),
            int(self.reference[self.ref_origin].shape[1] / 2 + x_size / 2),
        )

        dist_before_center = []
        dist_after_center = []
        for bead in range(0, len(mov_coors)):
            if (y_lim[1] > mov_coors[bead][0]) & (mov_coors[bead][0] > y_lim[0]):
                if (x_lim[1] > mov_coors[bead][1]) & (mov_coors[bead][1] > x_lim[0]):
                    dist_before_center.append(
                        distance.euclidean(mov_coors[bead], ref_coors[bead])
                    )
                    dist_after_center.append(
                        distance.euclidean(mov_transformed_coors[bead], ref_coors[bead])
                    )
        average_before_center = sum(dist_before_center) / len(dist_before_center)
        average_after_center = sum(dist_after_center) / len(dist_after_center)

        if (average_after_center - average_before_center) < 5:
            transform_qc = True

        return transform_qc, (average_after_center - average_before_center)

    def report_changes_in_mse(self) -> Tuple[bool, float]:
        """
        Report changes in normalized root mean-squared-error value before and after transform, post-segmentation.
        :return:
            qc: A boolean to indicate if it passed (True) or failed (False) qc
            diff_mse: Difference in mean squared error
        """
        qc = False

        mse_before = metrics.mean_squared_error(self.reference_seg, self.moving_seg)
        mse_after = metrics.mean_squared_error(
            self.reference_seg, self.moving_seg_transformed
        )

        diff_mse = mse_after - mse_before
        if diff_mse <= 0:
            qc = True

        return qc, diff_mse
