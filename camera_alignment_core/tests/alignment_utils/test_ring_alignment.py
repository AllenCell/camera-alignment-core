import typing

import numpy
import pandas
import pytest

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
