import dataclasses
import itertools
import math
import pathlib
import typing

from aicsimageio import AICSImage

from .constants import CameraPosition
from .exception import IncompatibleImageException


@dataclasses.dataclass(
    frozen=True,  # fields may not be assigned to after instance creation
)
class Channel:
    channel_index: int
    channel_name: str
    emission_wavelength: typing.Optional[float]
    camera_name: str
    camera_position: CameraPosition


class ChannelInfo:
    """
    This value object encapsulates behavior for working with channels within an image.
    """

    def __init__(
        self, image: AICSImage, image_path: typing.Union[str, pathlib.Path]
    ) -> None:
        self._image = image
        self._image_path = pathlib.Path(image_path)
        self._channels: typing.List[Channel] = []

    @property
    def channels(self) -> typing.List[Channel]:
        """ "List of channels, in order, within `image`.

        Note
        ----
        The current method for parsing channels from embedded, microscopy metadata is entirely
        dependent on the image being in CZI format.

        To support other formats, consider refactoring this utility class into a factory
        that constructs ChannelInfo's based on format, or better, consider parsing
        the embedded metadata in its OME form. The latter wasn't done because the mapping
        between OME XML and CZI XML is not yet reliable (as of 2021-11).
        """
        if not self._channels:
            if not self._is_czi_file():
                raise IncompatibleImageException(
                    f"ChannelInfo is only compatible with CZI images: cannot work with {self._image_path.name}"
                )

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

    def find_channels_closest_in_emission_wavelength_across_cameras(
        self,
    ) -> typing.Tuple[Channel, Channel]:
        """Return pair of Channels from separate cameras that are closest in their emission wavelength.

        Returns
        -------
        tuple[Channel, Channel] : A two-element tuple of Channel.
            Each Channel was acquired on a different camera, and among all the channels taken on each of the cameras,
            these are the channels that are closest in their emission wavelengths.

        Raises
        ------
        ValueError :
            If `image` was acquired using only one camera, its Channel(s) will be from the same camera, and therefore
            calling this method does not make sense.
            If somehow more than two cameras are used, the core logic of this method would break, and is not
            supported.
        """
        channels_sorted_by_camera = sorted(
            self.channels, key=lambda channel: channel.camera_name
        )
        unique_cameras = set()
        channels_grouped_by_camera: typing.List[typing.List[Channel]] = []
        for camera_name, channels_taken_by_camera in itertools.groupby(
            channels_sorted_by_camera, lambda channel: channel.camera_name
        ):
            unique_cameras.add(camera_name)
            channels_grouped_by_camera.append(list(channels_taken_by_camera))

        if len(channels_grouped_by_camera) != 2:
            raise ValueError(
                f"Expected 2 cameras, found {len(channels_grouped_by_camera)}: {unique_cameras}."
            )

        # Create cartesian product of channels from each camera
        channel_pairs = itertools.product(*channels_grouped_by_camera)

        # Determine pair of channels from separate cameras that have the minimum distance in their emission wavelengths
        def comparator(channel_tuple: tuple[Channel, Channel]):
            channel_a, channel_b = channel_tuple
            if not channel_a.emission_wavelength or not channel_b.emission_wavelength:
                return math.inf

            return abs(channel_a.emission_wavelength - channel_b.emission_wavelength)

        return min(channel_pairs, key=comparator)

    def _is_czi_file(self) -> bool:
        with open(self._image_path, "rb") as fd:
            return fd.read(10) == b"ZISRAWFILE"
