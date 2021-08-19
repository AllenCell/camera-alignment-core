import typing

from .constants import Channel


class ChannelInfo:
    """Abstraction for working with channels within an image"""

    def __init__(self, channel_to_index_map: typing.Dict[Channel, int]) -> None:
        self._channel_to_index_map = channel_to_index_map

    def channel_at_index(self, target_index: int) -> Channel:
        for channel, index in self._channel_to_index_map.items():
            if index == target_index:
                return channel

        raise IndexError(f"No channel exists at index {target_index}")

    def index_of_channel(self, Channel) -> int:
        return self._channel_to_index_map[Channel]

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ChannelInfo):
            return False

        return self._channel_to_index_map == o._channel_to_index_map

    def __iter__(self) -> typing.Iterator:
        """Support iteration over _channel_to_index_map items"""
        return iter(self._channel_to_index_map.items())

    def __len__(self) -> int:
        """
        The length of a ChannelInfo is tied to the length of it's private _channel_to_index_map.
        By extension, when _channel_to_index_map is empty, a ChannelInfo object will be considered
        to be False in a Boolean context (i.e. `bool(ChannelInfo({})) == False`)
        """
        return len(self._channel_to_index_map)

    def __repr__(self) -> str:
        return f"ChannelInfo({repr(self._channel_to_index_map)})"

    def __str__(self) -> str:
        return str(self._channel_to_index_map)
