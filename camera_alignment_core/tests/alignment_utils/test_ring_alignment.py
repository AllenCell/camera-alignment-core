import typing
from typing import Dict, List

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
            ((15, 10), 30, -20),
            ((15, 10), 60, -40),  # total perturb close to distance between pts
        ],
    )
    def test_assign_ref_to_mov(
        self, num_beads: typing.Tuple[int, int], perturb_x: int, perturb_y: int
    ):
        # Assign
        ref_dict = {}
        mov_dict = {}
        ref_data_dict: Dict[str, List[int]] = {
            "label": [],
            "centroid-0": [],
            "centroid-1": [],
        }
        mov_data_dict: Dict[str, List[int]] = {
            "label": [],
            "centroid-0": [],
            "centroid-1": [],
        }
        bead_num = 0
        for x in range(num_beads[0]):
            for y in range(num_beads[1]):
                ref_coor = (x * 100, y * 100)
                mov_coor = (ref_coor[0] + perturb_x, ref_coor[1] + perturb_y)

                ref_dict[bead_num] = ref_coor
                mov_dict[bead_num] = mov_coor

                ref_data_dict["label"].append(bead_num)
                ref_data_dict["centroid-0"].append(x * 100)
                ref_data_dict["centroid-1"].append(y * 100)
                mov_data_dict["label"].append(bead_num)
                mov_data_dict["centroid-0"].append(ref_coor[0] + perturb_x)
                mov_data_dict["centroid-1"].append(ref_coor[1] + perturb_y)
                bead_num += 1

        # Act
        # dummy init parameters
        ref_mov_coor_dict = RingAlignment(
            pandas.DataFrame(ref_data_dict),
            0,
            pandas.DataFrame(mov_data_dict),
            0,
        ).assign_ref_to_mov(ref_dict, mov_dict)

        missmatch = []
        for ref_coor, mov_coor in ref_mov_coor_dict.items():
            ref_bead = list(ref_dict.values()).index(ref_coor)
            mov_bead = list(mov_dict.values()).index(mov_coor)

            missmatch.append(not (ref_bead == mov_bead))

        # Assert
        assert not any(missmatch)
