import typing

import pytest

from camera_alignment_core.alignment_utils import (
    CropRings,
)


class CropDimensionsInput(typing.NamedTuple):
    img_height: int
    img_width: int
    cross_y: int
    cross_x: int
    bead_dist_px: float  # um


class CropDimensionsOutput(typing.NamedTuple):
    crop_top: int
    crop_bottom: int
    crop_left: int
    crop_right: int


class TestCropRings:
    @pytest.mark.parametrize(
        ["input_args", "expected"],
        [
            (
                CropDimensionsInput(
                    img_height=100,
                    img_width=200,
                    cross_y=60,  # slightly out of center
                    cross_x=105,  # slightly out of center
                    bead_dist_px=15,
                ),
                CropDimensionsOutput(
                    crop_top=8, crop_bottom=100, crop_left=8, crop_right=188
                ),
            ),
            (
                CropDimensionsInput(
                    img_height=100,
                    img_width=200,
                    cross_y=50,  # centered
                    cross_x=100,  # centered
                    bead_dist_px=15,
                ),
                CropDimensionsOutput(
                    crop_top=12, crop_bottom=88, crop_left=0, crop_right=200
                ),
            ),
            (
                CropDimensionsInput(
                    img_height=100,
                    img_width=200,
                    cross_y=50,  # centered
                    cross_x=100,  # centered
                    bead_dist_px=10,
                ),
                CropDimensionsOutput(
                    crop_top=5, crop_bottom=95, crop_left=5, crop_right=195
                ),
            ),
        ],
    )
    def test_get_crop_dimensions(
        self, input_args: CropDimensionsInput, expected: CropDimensionsOutput
    ):
        # Arrange / Act
        crop_top, crop_bottom, crop_left, crop_right = CropRings.get_crop_dimensions(
            img_height=input_args.img_height,
            img_width=input_args.img_width,
            cross_y=input_args.cross_y,
            cross_x=input_args.cross_x,
            bead_dist_px=input_args.bead_dist_px,
        )

        # Assert
        assert crop_top == expected.crop_top
        assert crop_bottom == expected.crop_bottom
        assert crop_left == expected.crop_left
        assert crop_right == expected.crop_right
