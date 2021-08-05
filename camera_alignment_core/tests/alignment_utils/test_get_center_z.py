import typing

import numpy
import pytest

from camera_alignment_core.alignment_utils import (
    get_center_z,
)


def generate_pseudo_image(target_z: int, offset_from_center: int):
    assert abs(offset_from_center) < target_z
    center = target_z - offset_from_center

    pseudo_image = (
        numpy.random.normal(loc=1, scale=0.1, size=(center * 2, 100, 100)) * 100
    )

    # rand_mask = numpy.random.normal(
    #     loc=0, scale=0.2, size=(100, 100)
    # ) >= 0
    # target_z_slice = numpy.zeros((100,100), )
    # pseudo_image[target_z][rand_mask] = numpy.random.normal(
    #     loc=.5, scale=.2, size=(100, 100)
    # ) *100
    pseudo_image[target_z, ...] = (
        numpy.random.normal(loc=1, scale=0.5, size=(100, 100)) * 100
    )

    return pseudo_image


class TestGetCenterZ:
    @pytest.mark.parametrize(
        ["target_z", "offset_from_center", "thresh"],
        [
            (4, 1, (0.2, 99.8)),
            (10, -2, (10, 80)),
            (6, 0, (5, 90)),
        ],
    )
    def test_get_center_z(
        self, target_z: int, offset_from_center: int, thresh: typing.Tuple[float, float]
    ):
        # Arrange
        test_image = generate_pseudo_image(target_z, offset_from_center)

        # Act
        center_z = get_center_z(test_image, thresh)

        # Assert
        assert target_z == center_z
