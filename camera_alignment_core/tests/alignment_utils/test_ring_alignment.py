import typing

import numpy
import pandas
import pytest
from skimage.transform import SimilarityTransform

from camera_alignment_core.alignment_utils import (
    RingAlignment,
)


class TestRingAlign:
    @pytest.mark.parametrize(
        ["num_beads", "perturb_x", "perturb_y"],
        [
            ((4, 4), 1, 1),
            ((10, 10), -4, 20),
            ((15, 10), 60, -40),
        ],
    )
    def test_assign_ref_to_mov(
        self, num_beads: typing.Tuple[int, int], perturb_x: int, perturb_y: int
    ):
        # Assign
        ref_dict = {}
        mov_dict = {}

        bead_num = 0
        for x in range(num_beads[0]):
            for y in range(num_beads[1]):
                ref_coor = (x * 100, y * 100)
                mov_coor = (ref_coor[0] + perturb_x, ref_coor[1] + perturb_y)

                ref_dict[bead_num] = ref_coor
                mov_dict[bead_num] = mov_coor
                bead_num += 1

        # Act
        # dummy init parameters
        ref_mov_coor_dict = RingAlignment(
            numpy.zeros((1)),
            numpy.zeros((1)),
            pandas.DataFrame(),
            0,
            numpy.zeros((1)),
            numpy.zeros((1)),
            pandas.DataFrame(),
            0,
        ).assign_ref_to_mov(ref_dict, mov_dict)

        missmatch = []
        for ref_coor, mov_coor in ref_mov_coor_dict.items():
            ref_bead = list(ref_dict.values()).index(ref_coor)
            mov_bead = list(mov_dict.values()).index(mov_coor)

            missmatch.append(not (ref_bead == mov_bead))

        # Assert
        assert not any(missmatch)

    def test_report_matrix_parameters(self):
        tform_matrix = numpy.array(
            [
                [
                    1.001828593258253797e00,
                    -5.167305751106508228e-03,
                    -5.272046139691610733e-02,
                ],
                [
                    5.167305751106508228e-03,
                    1.001828593258254241e00,
                    -3.061419473755620402e00,
                ],
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.000000000000000000e00,
                ],
            ]
        )

        tform = SimilarityTransform(matrix=tform_matrix)
        alignInfo = RingAlignment(
            numpy.zeros((1)),
            numpy.zeros((1)),
            pandas.DataFrame(),
            0,
            numpy.zeros((1)),
            numpy.zeros((1)),
            pandas.DataFrame(),
            0,
        ).report_similarity_matrix_parameters(tform, False)

        assert tform.scale == alignInfo.scaling
        assert tform.translation[0] == alignInfo.shift_y
        assert tform.translation[1] == alignInfo.shift_x
        assert tform.rotation == alignInfo.rotation
