import math
from random import randint
from typing import Any, Tuple

import numpy
from numpy.typing import NDArray
import pytest
from scipy.ndimage import gaussian_filter, shift
from skimage.morphology import ball
from skimage.transform import SimilarityTransform

from camera_alignment_core.alignment_utils.alignment_qc import (
    AlignmentQC,
)


def create_test_img_set(
    img_shape: Tuple[int, int] = (405, 611),
    offset: list[int] = [0, 0, 0],
    ref_intensity: int = 6000,
    moving_intensity: int = 6000,
    blur: bool = True,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:

    # Arrange
    ball_radius = 3

    ref = numpy.zeros((11,) + img_shape, numpy.uint16)
    sphere_template = ball(ball_radius)
    for y in numpy.arange(10, img_shape[0], 30):
        for x in numpy.arange(10, img_shape[1], 30):
            if (img_shape[1] - x < sphere_template.shape[2]) or (
                img_shape[0] - y < sphere_template.shape[1]
            ):
                continue

            ref[
                6 - ball_radius : 6 + ball_radius + 1,
                y - ball_radius : y + ball_radius + 1,
                x - ball_radius : x + ball_radius + 1,
            ] = sphere_template

    cross = numpy.zeros((11, 31, 31))
    cross[:, 13:20, :] = 1
    cross[:, :, 13:20] = 1
    cross[:2, :, :] = 0
    cross[-2:, :, :] = 0

    ref[:, 174:205, 294:325] = cross
    moving = numpy.copy(ref)

    ref_seg = ref
    moving_seg = moving

    moving = shift(moving, offset, order=0).astype(numpy.uint16) > 0

    ref_raw = ref + ref_intensity // 5
    moving_raw = moving + moving_intensity // 5

    ref_raw[ref > 0] = ref_intensity
    moving_raw[moving > 0] = moving_intensity

    if blur:
        ref_raw = gaussian_filter(ref_raw, 0.5, output=numpy.uint16)
        moving_raw = gaussian_filter(moving_raw, 0.5, output=numpy.uint16)

    return ref_raw, moving_raw, ref_seg, moving_seg


class TestAlignmentQC:
    @pytest.mark.parametrize(
        ["offset"],
        [
            ([2, 0, 0],),
            ([-2, 0, 0],),
            ([3, 5, -8],),
            ([1, 15, -7],),
            ([-6, 15, -7],),
        ],
    )
    def test_check_z_offset(self, offset):
        # Arrange
        qc = AlignmentQC()

        ref, moving, _, _ = create_test_img_set(offset=offset)
        qc.set_raw_images(ref, moving)
        qc.set_centers()

        # Act
        z_offset, ref_origin, mov_origin = qc.check_z_offset_between_ref_mov()

        # Assert
        assert z_offset == -offset[0]
        assert ref_origin == 6
        assert mov_origin == 6 + offset[0]

    @pytest.mark.parametrize(
        ["ref_intensity", "mov_intensity"],
        [
            (6000, 6000),
            (12000, 8000),
            (33000, 14000),
            (57000, 42000),
        ],
    )
    def test_image_snr(self, ref_intensity: int, mov_intensity: int):
        # Arrange
        qc = AlignmentQC()
        ref, moving, ref_seg, moving_seg = create_test_img_set(
            ref_intensity=ref_intensity, moving_intensity=mov_intensity, blur=False
        )

        qc.set_raw_images(ref, moving)
        qc.set_centers(ref_center=6, mov_center=6)
        qc.set_seg_images(ref_seg[6], moving_seg[6])

        # Act
        ref_signal, ref_noise, mov_signal, mov_noise = qc.report_ref_mov_image_snr()

        # Assert
        assert ref_signal == ref_intensity
        assert mov_signal == mov_intensity
        assert ref_noise == ref_intensity // 5
        assert mov_noise == mov_intensity // 5

    @pytest.mark.parametrize(
        ["num_beads", "qc_result"],
        [
            (10, True),
            (15, True),
            (5, False),
        ],
    )
    def test_report_number_beads(self, num_beads: int, qc_result: bool):
        # Arrange
        def XAND(a: bool, b: bool):
            if a and b:
                return True
            elif ~a and ~b:
                return True
            else:
                return False

        qc = AlignmentQC()
        bead_dict = {}
        for i in range(num_beads):
            bead_dict[(randint(0, 1900), randint(0, 800))] = (
                randint(0, 1900),
                randint(0, 800),
            )
        qc.set_ref_mov_coor_dict(bead_dict)

        # Act
        bead_num_qc, bead_count = qc.report_number_beads()

        # Assert
        assert bead_count == num_beads
        assert XAND(bead_num_qc, qc_result)

    @pytest.mark.parametrize(
        ["moving_intensity", "offset"],
        [
            (6000, [0, 0, 0]),
            (12000, [0, -2, 0]),
            (24000, [0, 5, -8]),
            (48000, [0, 15, -7]),
        ],
    )
    def test_report_change_fov_intensity_parameters(
        self, moving_intensity: int, offset: list[int]
    ):
        # Arrange
        ref_raw, moving_raw, ref_seg, moving_seg = create_test_img_set(
            offset=offset, moving_intensity=moving_intensity, blur=False
        )
        tform = SimilarityTransform(translation=[-o for o in offset[1:]])

        qc = AlignmentQC()
        qc.set_raw_images(ref_raw, moving_raw)
        qc.set_seg_images(ref_seg, moving_seg)
        qc.set_centers()
        qc.set_tform(tform)

        # Act
        param_dict = qc.report_change_fov_intensity_parameters()

        # Assert
        assert param_dict["median_intensity"] == 0
        assert (param_dict["min_intensity"] == -numpy.amin(moving_raw)) or (
            param_dict["min_intensity"] == 0
        )
        assert param_dict["max_intensity"] == 0
        assert (param_dict["min_intensity"] == -numpy.amin(moving_raw)) or (
            param_dict["1st_percentile"] == 0
        )
        assert param_dict["995th_percentile"] == 0

        return

    @pytest.mark.parametrize(
        ["img_shape", "offset"],
        [
            (
                (405, 611),
                [0, 0],
            ),
            (
                (405, 611),
                [2, 0],
            ),
            (
                (405, 611),
                [5, -8],
            ),
            (
                (405, 611),
                [15, -7],
            ),
            (
                (405, 611),
                [-3, 8],
            ),
        ],
    )
    def test_report_changes_in_coordinate_mapping(
        self, img_shape: Tuple[int, int], offset: list[int]
    ):
        # Arrange
        beads = {}
        for y in numpy.arange(10, img_shape[0], 30):
            for x in numpy.arange(10, img_shape[1], 30):
                beads[(y, x)] = (y + offset[0], x + offset[1])

        ref_raw, moving_raw, ref_seg, moving_seg = create_test_img_set(
            img_shape=img_shape, blur=False
        )

        tform = SimilarityTransform(translation=[-o for o in offset])

        qc = AlignmentQC()
        qc.set_raw_images(ref_raw, moving_raw)
        qc.set_seg_images(ref_seg, moving_seg)
        qc.set_centers()
        qc.set_tform(tform)
        qc.set_ref_mov_coor_dict(beads)

        # Act
        transform_qc, distance = qc.report_changes_in_coordinates_mapping()

        # Assert
        assert transform_qc
        assert distance <= math.sqrt(offset[0] ** 2 + offset[1] ** 2)

    @pytest.mark.parametrize(
        ["img_shape", "offset"],
        [
            (
                (405, 611),
                [2, 0],
            ),
            (
                (405, 611),
                [5, -8],
            ),
            (
                (405, 611),
                [15, -7],
            ),
            (
                (405, 611),
                [-3, 8],
            ),
        ],
    )
    def report_changes_in_mse(self, img_shape: Tuple[int, int], offset: list[int]):
        # Arrange
        ref_raw, moving_raw, ref_seg, moving_seg = create_test_img_set(
            img_shape=img_shape, blur=False
        )

        tform = SimilarityTransform(translation=[-o for o in offset])

        qc = AlignmentQC()
        qc.set_raw_images(ref_raw, moving_raw)
        qc.set_seg_images(ref_seg, moving_seg)
        qc.set_centers()
        qc.set_tform(tform)

        # Act
        qc_result, diff_mse = qc.report_changes_in_mse()

        # Assert
        assert qc_result
