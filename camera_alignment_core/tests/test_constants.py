import pytest

from camera_alignment_core.constants import (
    CameraPosition,
)


@pytest.mark.parametrize(
    ["detector_name", "expected"],
    [
        ("", CameraPosition.LEFT),
        ("", CameraPosition.BACK),
        pytest.param("Fake", None, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_camera_position_from_czi_detector_name(
    detector_name: str, expected: CameraPosition
) -> None:
    # Arrange / Act
    actual = CameraPosition.from_czi_detector_name(detector_name)

    # Assert
    assert actual == expected
