import pytest

from camera_alignment_core.constants import (
    Channel,
)


@pytest.mark.parametrize(
    ["nominal_magnification", "expected"],
    [
        (405, Channel.RAW_405_NM),
        (488, Channel.RAW_488_NM),
        (561, Channel.RAW_561_NM),
        (638, Channel.RAW_638_NM),
        pytest.param(999, None, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_channel_from_magnification(
    nominal_magnification: int, expected: Channel
) -> None:
    # Arrange / Act
    actual = Channel.from_magnification(nominal_magnification)

    # Assert
    assert actual == expected
