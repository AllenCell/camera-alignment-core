import pathlib
import typing

from aicsimageio import AICSImage

from ..exception import IncompatibleImageException
from .channel_info_abc import (
    CameraPosition,
    Channel,
    ChannelInfo,
)
from .czi_channel_info import CziChannelInfo


def channel_info_factory(image_path: typing.Union[str, pathlib.Path]) -> ChannelInfo:
    """Construct a concrete `ChannelInfo` instance that is type-appropriate for a given image.

    Current concrete `ChannelInfo` implementations:
        1. CziChannelInfo, supporting CZI images.
    """
    if CziChannelInfo.is_czi_file(image_path):
        return CziChannelInfo(AICSImage(image_path))

    error_msg = (
        f"Unable to instantiate a ChannelInfo for {pathlib.Path(image_path).name}. "
        "Only CZI images are currently supported."
    )
    raise IncompatibleImageException(error_msg)


__all__ = ("channel_info_factory", "CameraPosition", "Channel", "ChannelInfo")
