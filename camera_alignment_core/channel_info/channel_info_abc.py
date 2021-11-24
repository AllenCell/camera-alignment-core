import abc
import dataclasses
import enum
import itertools
import math
import typing

from aicsimageio import AICSImage


class CameraPosition(enum.Enum):
    LEFT = "Left"
    BACK = "Back"

    @staticmethod
    def from_czi_detector_name(detector_name: str) -> "CameraPosition":
        """Given a detector name like 'Detector:Camera 2 (Left)' and 'Detector:Camera 1 (Back)',
        return CameraPosition that best matches.

        This is incredibly heuristic driven, but holds up reliably across tests of CZI images
        acquired at AICS since 2018/2019ish.

        Parameters
        ----------
        detector_name : str
            This corresponds to the "Id" attribute from `InstrumentDetector` XML elements from
            embedded metadata within CZI images. Example XML path:
                Directly: "Metadata/Information/Instrument/Detectors"
                Via channels: "Metadata/Information/Image/Dimensions/Channels/DetectorSettings/Detector"
        """
        lower_cased_detector = detector_name.lower()
        if CameraPosition.LEFT.value.lower() in lower_cased_detector:
            return CameraPosition.LEFT
        if CameraPosition.BACK.value.lower() in lower_cased_detector:
            return CameraPosition.BACK

        raise ValueError(
            f"Cannot find appropriate CameraPosition for detector: {detector_name}"
        )


@dataclasses.dataclass(
    frozen=True,  # fields may not be assigned to after instance creation
)
class Channel:
    channel_index: int
    channel_name: str
    emission_wavelength: typing.Optional[float]
    camera_name: str
    camera_position: CameraPosition


class ChannelInfo(abc.ABC):
    """This value object encapsulates behavior for querying for channels within an image.

    Create a ChannelInfo using the `channel_info_factory` factory function exported from
    the channel_info module. That factory will provide a concrete class implementing this
    interface that is appropriate for a given image format.
    """

    def __init__(
        self,
        image: AICSImage,
    ) -> None:
        self._image = image
        self._channels: typing.List[Channel] = []

    @abc.abstractproperty
    def channels(self) -> typing.List[Channel]:
        """List of channels, in order, within `image`."""
        pass

    def channels_from_camera_position(
        self, camera_position: CameraPosition
    ) -> typing.List[Channel]:
        """Return listing of Channels that were acquired on the camera with position `camera_position`.

        Example
        -------
        Get list of all back-camera channels
        >>> channel_info = channel_info_factory("/path/to/image.czi")
        >>> back_camera_channels = channel_info.channels_from_camera_position(CameraPosition.BACK)
        """
        return [
            channel
            for channel in self.channels
            if channel.camera_position == camera_position
        ]

    def find_channels_closest_in_emission_wavelength_between_cameras(
        self,
    ) -> typing.Tuple[Channel, Channel]:
        """Return pair of Channels from separate cameras that are closest in their emission wavelength.

        This is intended to be used to dynamically choose the two channels of an optical control image
        that should be used to generate an alignment matrix. As such, this method only has
        expected utility if the image passed to `ChannelInfo` is an optical control.

        Returns
        -------
        Tuple[Channel, Channel] : A two-element tuple of Channel.
            Each Channel was acquired on a different camera, and among all the channels taken on each of the cameras,
            these are the channels that are closest in their emission wavelengths. Returned Channels are sorted in ascending
            order by their emission wavelength.

        Raises
        ------
        ValueError :
            If `image` was acquired using only one camera, its Channel(s) will be from the same camera, and therefore
            calling this method does not make sense. If somehow more than two cameras are used, the core logic of this method would break,
            and is likewise not supported.
        """
        channels_sorted_by_camera = sorted(
            self.channels, key=lambda channel: channel.camera_name
        )

        channels_grouped_by_camera: typing.List[typing.List[Channel]] = []
        for _, channels_taken_by_camera in itertools.groupby(
            channels_sorted_by_camera, lambda channel: channel.camera_name
        ):
            channels_grouped_by_camera.append(list(channels_taken_by_camera))

        if len(channels_grouped_by_camera) != 2:
            unique_cameras = set()
            for channel_group in channels_grouped_by_camera:
                unique_cameras.add(channel_group[0].camera_name)
            raise ValueError(
                f"Expected 2 cameras, found {len(channels_grouped_by_camera)}: {unique_cameras}."
            )

        # Create cartesian product of channels from each camera, pairing each channel from each camera with each other
        # E.g.: [(channel_a_from_cam_1, channel_a_from_cam_2), (channel_a_from_cam_1, channel_b_from_cam_2), ...]
        channel_pairs = itertools.product(
            channels_grouped_by_camera[0], channels_grouped_by_camera[1]
        )

        # Determine pair of channels from separate cameras that have the minimum distance in their emission wavelengths
        def comparator(channel_tuple: typing.Tuple[Channel, Channel]) -> float:
            channel_a, channel_b = channel_tuple
            if not channel_a.emission_wavelength or not channel_b.emission_wavelength:
                return math.inf

            return abs(channel_a.emission_wavelength - channel_b.emission_wavelength)

        min_pairing = min(channel_pairs, key=comparator)
        channel_a, channel_b = sorted(
            min_pairing,
            key=lambda channel: channel.emission_wavelength
            if channel.emission_wavelength is not None
            else math.inf,
        )
        return (channel_a, channel_b)
