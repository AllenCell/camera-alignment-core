import typing

import pytest

from camera_alignment_core.channel_info import (
    CameraPosition,
    Channel,
    channel_info_factory,
)
from camera_alignment_core.channel_info.czi_channel_info import (
    CziChannelInfo,
)

from . import (
    ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
    GENERIC_CZI_URL,
    GENERIC_OME_TIFF_URL,
    UNALIGNED_ZSD1_IMAGE_URL,
    get_test_image,
)


@pytest.mark.parametrize(
    ["detector_name", "expected"],
    # GM: All of the detector names were taken from a sampling of real
    # data stored in FMS (1153 CZI images sampled Nov. 2021)
    [
        ("Detector:Camera 2 (Left)", CameraPosition.LEFT),
        ("Detector:Camera 2 (left)", CameraPosition.LEFT),
        ("Detector:Camera 2 Left", CameraPosition.LEFT),
        ("Detector:Hammamatsu Left", CameraPosition.LEFT),
        ("Detector:ORCA left", CameraPosition.LEFT),
        ("Detector:Orca Left", CameraPosition.LEFT),
        ("Detector:Camera 1 (Back)", CameraPosition.BACK),
        ("Detector:Camera 1 (back)", CameraPosition.BACK),
        pytest.param("Detector:0:1", None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(
            "Detector: LSM800 GaAsP-Pmt1",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            "Detector:Hamamatsu Camera",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_camera_position_from_czi_detector_name(
    detector_name: str, expected: CameraPosition
) -> None:
    # Arrange / Act
    actual = CameraPosition.from_czi_detector_name(detector_name)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    ["image_url", "camera_position"],
    [
        (UNALIGNED_ZSD1_IMAGE_URL, CameraPosition.BACK),
        (UNALIGNED_ZSD1_IMAGE_URL, CameraPosition.LEFT),
    ],
)
def test_channel_info_channels_from_camera_position(
    image_url: str, camera_position: CameraPosition
) -> None:
    # Arrange
    _, image_path = get_test_image(image_url)
    channel_info = channel_info_factory(image_path)

    # Act
    channels_from_position = channel_info.channels_from_camera_position(camera_position)

    # Assert
    assert channels_from_position  # assert non-empty
    assert all(
        [
            channel.camera_position == camera_position
            for channel in channels_from_position
        ]
    )


@pytest.mark.parametrize(
    ["image_url", "expected"],
    [
        (
            UNALIGNED_ZSD1_IMAGE_URL,
            [
                Channel(
                    channel_index=0,
                    channel_name="Bright_2",
                    emission_wavelength=None,
                    camera_name="Detector:Camera 1 (Back)",
                    camera_position=CameraPosition.BACK,
                ),
                Channel(
                    channel_index=1,
                    channel_name="EGFP",
                    emission_wavelength=509.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
                Channel(
                    channel_index=2,
                    channel_name="CMDRP",
                    emission_wavelength=676.0,
                    camera_name="Detector:Camera 1 (Back)",
                    camera_position=CameraPosition.BACK,
                ),
                Channel(
                    channel_index=3,
                    channel_name="H3342",
                    emission_wavelength=455.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
            ],
        ),
        (
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
            [
                Channel(
                    channel_index=0,
                    channel_name="Bright_2",
                    emission_wavelength=None,
                    camera_name="Detector:Camera 1 (Back)",
                    camera_position=CameraPosition.BACK,
                ),
                Channel(
                    channel_index=1,
                    channel_name="EGFP",
                    emission_wavelength=509.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
                Channel(
                    channel_index=2,
                    channel_name="TaRFP",
                    emission_wavelength=583.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
                Channel(
                    channel_index=3,
                    channel_name="CMDRP",
                    emission_wavelength=676.0,
                    camera_name="Detector:Camera 1 (Back)",
                    camera_position=CameraPosition.BACK,
                ),
                Channel(
                    channel_index=4,
                    channel_name="H3342",
                    emission_wavelength=455.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
            ],
        ),
    ],
)
def test_channels(image_url: str, expected: typing.List[Channel]):
    # Arrange
    _, image_path = get_test_image(image_url)

    channel_info = channel_info_factory(image_path)

    # Act / Assert
    assert channel_info.channels == expected


@pytest.mark.parametrize(
    ["image_url", "expected"],
    [
        (
            UNALIGNED_ZSD1_IMAGE_URL,
            (
                Channel(
                    channel_index=1,
                    channel_name="EGFP",
                    emission_wavelength=509.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
                Channel(
                    channel_index=2,
                    channel_name="CMDRP",
                    emission_wavelength=676.0,
                    camera_name="Detector:Camera 1 (Back)",
                    camera_position=CameraPosition.BACK,
                ),
            ),
        ),
        (
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
            (
                Channel(
                    channel_index=2,
                    channel_name="TaRFP",
                    emission_wavelength=583.0,
                    camera_name="Detector:Camera 2 (Left)",
                    camera_position=CameraPosition.LEFT,
                ),
                Channel(
                    channel_index=3,
                    channel_name="CMDRP",
                    emission_wavelength=676.0,
                    camera_name="Detector:Camera 1 (Back)",
                    camera_position=CameraPosition.BACK,
                ),
            ),
        ),
    ],
)
def test_channel_info_find_channels_closest_in_emission_wavelength_between_cameras(
    image_url: str, expected: typing.Tuple[Channel, Channel]
):
    # Arrange
    _, image_path = get_test_image(image_url)

    channel_info = channel_info_factory(image_path)

    # Act
    actual = channel_info.find_channels_closest_in_emission_wavelength_between_cameras()

    # Assert
    assert actual == expected

    # assert that the Channels are sorted by emission_wavelength
    channel_a, channel_b = actual
    assert channel_a.emission_wavelength
    assert channel_b.emission_wavelength
    assert channel_a.emission_wavelength < channel_b.emission_wavelength


@pytest.mark.parametrize(
    ["image_url", "expected"],
    [
        (GENERIC_CZI_URL, True),
        (UNALIGNED_ZSD1_IMAGE_URL, True),
        (GENERIC_OME_TIFF_URL, False),
    ],
)
def test_czi_channel_info_is_czi_file(image_url: str, expected: bool) -> None:
    # Arrange
    _, image_path = get_test_image(image_url)

    # Act
    actual = CziChannelInfo.is_czi_file(image_path)

    # Assert
    assert actual == expected
