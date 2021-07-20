import logging
import typing

from aicsimageio import AICSImage
from matplotlib.pyplot import imsave
import numpy
import pytest
from skimage.transform import (
    SimilarityTransform,
    warp,
)

from camera_alignment_core import AlignmentCore
from camera_alignment_core.constants import (
    LOGGER_NAME,
)

log = logging.getLogger(LOGGER_NAME)


ZSD_100x_OPTICAL_CONTROL_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_ZSD1_100X_SLF-015_20210624.czi"  # noqa E501

# FMS ID: 0023c446cd384dc3947c90dc7a76f794; 303.38 MB
GENERIC_OME_TIFF_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/3500003897_100X_20200306_1r-Scene-30-P89-G11.ome.tiff"  # noqa E501

# FMS ID: 439c852ea76e46d4b9a9f8813f331b4d; 264.43 MB
GENERIC_CZI_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/20200701_N04_001.czi"  # noqa E501


class TestAlignmentCore:
    def setup_method(self):
        """You can use this to setup before each test"""
        self.alignment_core = AlignmentCore()

    # @pytest.mark.skip("AlignmentCore::generate_alignment_matrix not yet implemented")
    def test_generate_alignment_matrix(
        self,
        get_image: typing.Callable[[str], AICSImage],
        caplog: pytest.LogCaptureFixture,
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
        expected_matrix = numpy.array(
            [
                [
                    1.001828593258253797e00,
                    -5.167305751106508228e-03,
                    -5.272046139691610733e-02,
                ],
                [
                    5.167305751106508228e-03,
                    1.001828593258254241e00,
                    -3.061419473755620402e00,
                ],
                [0, 0, 1],
            ]
        )

        expected_transform = SimilarityTransform(matrix=expected_matrix)

        for z in range(optical_control_image_data.shape[1]):
            optical_control_image_data[2, z, ...] = warp(
                optical_control_image_data[1, z, ...],
                expected_transform,
                preserve_range=True,
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

        matDet = numpy.linalg.det(
            numpy.matmul(numpy.array(actual_alignment_matrix.params).T, expected_matrix)
        )
        log.debug("Determinant (should be 1)")
        log.debug(matDet)

        assert abs(matDet - 1) < 0.0005

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
