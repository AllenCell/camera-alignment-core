import pathlib
import typing

from .channel_info_abc import (
    CameraPosition,
    Channel,
    ChannelInfo,
)


class CziChannelInfo(ChannelInfo):
    """ChannelInfo value object specific to CZI images. See docstrings in ChannelInfo.

    Notes
    -----
    This specialization of ChannelInfo is required because the CZI -> OME metadata conversion is
    not yet (as of 2021-11) reliable. Once that conversion is production-ready and Channels can be
    reliably constituted from OME metadata, this specialization can be safely deleted.
    """

    @staticmethod
    def is_czi_file(image_path: typing.Union[str, pathlib.Path]) -> bool:
        with open(image_path, "rb") as fd:
            return fd.read(10) == b"ZISRAWFILE"

    @property
    def channels(self) -> typing.List[Channel]:
        if not self._channels:
            channels: typing.List[Channel] = []
            for channel_index, channel in enumerate(
                self._image.metadata.find(
                    "Metadata/Information/Image/Dimensions/Channels"
                )
            ):
                emission_wavelength = channel.find("EmissionWavelength")
                detector = channel.find("DetectorSettings/Detector")
                detector_name = detector.attrib.get("Id")
                parsed_wavelength = (
                    float(emission_wavelength.text) if emission_wavelength else None
                )
                channels.append(
                    Channel(
                        channel_index,
                        channel.attrib.get("Name"),
                        parsed_wavelength,
                        detector_name,
                        CameraPosition.from_czi_detector_name(detector_name),
                    )
                )

            self._channels = channels
        return self._channels
