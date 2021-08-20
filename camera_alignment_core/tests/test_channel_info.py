import typing

import pytest

from camera_alignment_core.channel_info import (
    ChannelInfo,
)
from camera_alignment_core.constants import (
    Channel,
)


@pytest.mark.parametrize(
    ["channel_to_index_map", "index", "expected"],
    [
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            0,
            Channel.RAW_405_NM,
        ),
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_BRIGHTFIELD: 2},
            2,
            Channel.RAW_BRIGHTFIELD,
        ),
        pytest.param(
            {Channel.RAW_405_NM: 0},
            3,
            None,
            marks=pytest.mark.xfail(raises=IndexError),
        ),
    ],
)
def test_channel_at_index(
    channel_to_index_map: typing.Dict[Channel, int], index: int, expected: Channel
) -> None:
    # Arrange
    channel_info = ChannelInfo(channel_to_index_map)

    # Act
    actual = channel_info.channel_at_index(index)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    ["channel_to_index_map", "channel", "expected"],
    [
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            Channel.RAW_405_NM,
            0,
        ),
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_BRIGHTFIELD: 2},
            Channel.RAW_BRIGHTFIELD,
            2,
        ),
        pytest.param(
            {Channel.RAW_405_NM: 0},
            Channel.RAW_BRIGHTFIELD,
            None,
            marks=pytest.mark.xfail(raises=KeyError),
        ),
    ],
)
def test_index_of_channel(
    channel_to_index_map: typing.Dict[Channel, int], channel: Channel, expected: int
) -> None:
    # Arrange
    channel_info = ChannelInfo(channel_to_index_map)

    # Act
    actual = channel_info.index_of_channel(channel)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    ["channel_to_index_map", "other", "expected"],
    [
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            ChannelInfo({Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1}),
            True,
        ),
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            ChannelInfo({Channel.RAW_405_NM: 0}),
            False,
        ),
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            False,
        ),
    ],
)
def test_equals(
    channel_to_index_map: typing.Dict[Channel, int], other: typing.Any, expected: bool
) -> None:
    # Arrange
    channel_info = ChannelInfo(channel_to_index_map)

    # Act
    actual = channel_info == other

    # Assert
    assert actual == expected


def test_iter() -> None:
    # Arrange
    channel_to_index_map = {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1}
    channel_info = ChannelInfo(channel_to_index_map)

    # Act / Assert
    assert list(channel_info) == list(channel_to_index_map.items())


@pytest.mark.parametrize(
    ["channel_to_index_map", "expected"],
    [
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1},
            2,
        ),
        (
            {Channel.RAW_405_NM: 0},
            1,
        ),
        (
            {Channel.RAW_405_NM: 0, Channel.RAW_488_NM: 1, Channel.RAW_638_NM: 2},
            3,
        ),
        (
            {},
            0,
        ),
    ],
)
def test_len(channel_to_index_map: typing.Dict[Channel, int], expected: int) -> None:
    # Arrange
    channel_info = ChannelInfo(channel_to_index_map)

    # Act / Assert
    assert len(channel_info) == expected
