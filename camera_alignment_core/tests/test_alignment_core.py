import logging
import typing

from aicsimageio import AICSImage
import numpy
import numpy.testing
import pytest

from camera_alignment_core import AlignmentCore
from camera_alignment_core.constants import (
    LOGGER_NAME,
)

log = logging.getLogger(LOGGER_NAME)


ZSD_100x_OPTICAL_CONTROL_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_ZSD1_100X_SLF-015_20210624.czi"

# FMS ID: 0023c446cd384dc3947c90dc7a76f794; 303.38 MB
GENERIC_OME_TIFF_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/3500003897_100X_20200306_1r-Scene-30-P89-G11.ome.tiff"  # noqa E501

# FMS ID: 439c852ea76e46d4b9a9f8813f331b4d; 264.43 MB
GENERIC_CZI_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/20200701_N04_001.czi"  # noqa E501

UNALIGNED_ZSD1_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/3500004473_100X_20210430_1c-Scene-24-P96-G06.czi"
ALIGNED_ZSD1_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/3500004473_100X_20210430_1c-alignV2-Scene-24-P96-G06.ome.tiff"
ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_100X_20210430_fieldofrings.czi"
ARGOLIGHT_ALIGNED_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_100X_20210430_fieldofrings_aligned.tif"


class TestAlignmentCore:
    def setup_method(self):
        """You can use this to setup before each test"""
        self.alignment_core = AlignmentCore()

    @pytest.mark.slow
    def test_generate_alignment_matrix(
        self,
        get_image: typing.Callable[[str], AICSImage],
        caplog: pytest.LogCaptureFixture,
        output: bool = False,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        zsd_100x_optical_control_image = get_image(ZSD_100x_OPTICAL_CONTROL_IMAGE_URL)
        optical_control_image_data = zsd_100x_optical_control_image.get_image_data(
            "CZYX"
        )
        reference_channel = 1
        shift_channel = 2
        magnification = 100
        pixel_size_xy = 0.108

        # This is the output of the old alignment code for the optical control saved at
        # /allen/aics/microscopy/PRODUCTION/OpticalControl/ArgoLight/Argo_QC_Daily/ZSD1/ZSD1_argo_100X_SLF-015_20210624

        # Original Matrix
        # expected_matrix = numpy.array(
        #     [
        #         [
        #             1.001828593258253797e00,
        #             -5.167305751106508228e-03,
        #             -5.272046139691610733e-02,
        #         ],
        #         [
        #             5.167305751106508228e-03,
        #             1.001828593258254241e00,
        #             -3.061419473755620402e00,
        #         ],
        #         [
        #             0.000000000000000000e00,
        #             0.000000000000000000e00,
        #             1.000000000000000000e00,
        #         ],
        #     ]
        # )

        expected_matrix = numpy.array(
            [
                [
                    1.00022523e00,
                    8.68926296e-05,
                    -2.10827503e-01,
                ],
                [
                    -8.68926296e-05,
                    1.00022523e00,
                    2.84065985e-02,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    1.00000000e00,
                ],
            ]
        )

        # Act
        (
            actual_alignment_matrix,
            actual_alignment_info,
        ) = self.alignment_core.generate_alignment_matrix(
            optical_control_image_data,
            reference_channel,
            shift_channel,
            magnification,
            pixel_size_xy,
        )

        # Assert
        log.debug("Expected Matrix")
        log.debug(expected_matrix)
        log.debug("Estimated Matrix")
        log.debug(numpy.array(actual_alignment_matrix.params))

        assert numpy.allclose(expected_matrix, actual_alignment_matrix.params)

    @pytest.mark.slow
    def test_generate_alignment_matrix_reproducability(
        self,
        get_image: typing.Callable[[str], AICSImage],
        caplog: pytest.LogCaptureFixture,
        output: bool = False,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        zsd_100x_optical_control_image = get_image(ZSD_100x_OPTICAL_CONTROL_IMAGE_URL)
        optical_control_image_data = zsd_100x_optical_control_image.get_image_data(
            "CZYX"
        )
        reference_channel = 1
        shift_channel = 2
        magnification = 100
        pixel_size_xy = 0.108

        # Act
        (alignment_matrix_1, _,) = self.alignment_core.generate_alignment_matrix(
            optical_control_image_data,
            reference_channel,
            shift_channel,
            magnification,
            pixel_size_xy,
        )
        (alignment_matrix_2, _,) = self.alignment_core.generate_alignment_matrix(
            optical_control_image_data,
            reference_channel,
            shift_channel,
            magnification,
            pixel_size_xy,
        )

        mat1 = numpy.array(alignment_matrix_1.params)
        mat2 = numpy.array(alignment_matrix_1.params)

        # Assert
        log.debug("First Estimated Matrix")
        log.debug(mat1)

        log.debug("Second Estimated Matrix")
        log.debug(mat2)

        assert numpy.allclose(mat1, mat2)

    @pytest.mark.parametrize(
        ["image_path", "expectation"],
        [
            (
                GENERIC_OME_TIFF_URL,
                {
                    "Raw brightfield": 0,
                    "Raw 488nm": 1,
                    "Raw 638nm": 2,
                    "Raw 405nm": 3,
                },
            ),
            (GENERIC_CZI_URL, {"Raw 561nm": 1}),
        ],
    )
    @pytest.mark.slow
    def test_get_channel_name_to_index_map(
        self,
        image_path,
        expectation,
        get_image: typing.Callable[[str], AICSImage],
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        image = get_image(image_path)

        # Act
        result = self.alignment_core.get_channel_name_to_index_map(image)

        # Assert
        assert result == expectation

    @pytest.mark.parametrize(
        [
            "image_path",
            "alignment_image_path",
            "expectation_image_path",
            "magnification",
        ],
        [
            (
                UNALIGNED_ZSD1_IMAGE_URL,
                ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
                ALIGNED_ZSD1_IMAGE_URL,
                100,
            ),
        ],
    )
    @pytest.mark.slow
    def test_align_image(
        self,
        image_path,
        alignment_image_path,
        expectation_image_path,
        magnification,
        get_image: typing.Callable[[str], AICSImage],
        caplog: pytest.LogCaptureFixture,
    ):

        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        image = get_image(image_path)
        optical_control_image = get_image(alignment_image_path).get_image_data(
            "CZYX", T=0
        )
        channels_to_align = self.alignment_core.get_channel_name_to_index_map(image)
        (
            alignment_matrix,
            alignment_info,
        ) = self.alignment_core.generate_alignment_matrix(
            optical_control_image=optical_control_image,
            shift_channel=channels_to_align["Raw 638nm"],
            reference_channel=channels_to_align["Raw 405nm"],
            magnification=magnification,
            px_size_xy=optical_control_image.physical_pixel_sizes.X,
        )

        expectation = get_image(expectation_image_path).get_image_data("CZYX", T=0)

        # Act
        result = self.alignment_core.align_image(
            alignment_matrix=alignment_matrix,
            image=image.get_image_data("CZYX", T=0),
            channels_to_align=channels_to_align,
            magnification=magnification,
        )

        # Assert
        assert numpy.allclose(result, expectation)

    # TODO: Add 63x and 20x images to test
    @pytest.mark.parametrize(
        ["image_path", "magnification", "expected_shape"],
        [
            (
                UNALIGNED_ZSD1_IMAGE_URL,
                100,
                (4, 75, 600, 900),
            ),
        ],
    )
    def test_crop(
        self,
        image_path,
        magnification,
        expected_shape,
        get_image: typing.Callable[[str], AICSImage],
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        image = get_image(image_path).get_image_dask_data()[0]

        result = self.alignment_core._crop(image, magnification)

        assert result.shape == expected_shape
