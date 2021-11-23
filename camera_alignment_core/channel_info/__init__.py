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


def create_channel_info(image_path: typing.Union[str, pathlib.Path]) -> ChannelInfo:
    if CziChannelInfo.is_czi_file(image_path):
        return CziChannelInfo(AICSImage(image_path))

    error_msg = (
        f"Unable to instantiate a ChannelInfo for {pathlib.Path(image_path).name}. "
        "Only CZI images are currently supported."
    )
    raise IncompatibleImageException(error_msg)


__all__ = ("create_channel_info", "CameraPosition", "Channel", "ChannelInfo")
