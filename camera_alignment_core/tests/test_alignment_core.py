import logging
from os import mkdir
from os.path import exists
import typing

from aicsimageio import AICSImage
import numpy
import pytest
from skimage.io import imsave
from skimage.transform import (
    SimilarityTransform,
    warp,
)

from camera_alignment_core import AlignmentCore
from camera_alignment_core.constants import (
    LOGGER_NAME,
)

log = logging.getLogger(LOGGER_NAME)


ZSD_100x_OPTICAL_CONTROL_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_20210419_100X_ZSD1.czi"


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

        expected_matrix = numpy.array(
            [
                [
                    1.001096985136178619e00,
                    -4.788795753668498474e-03,
                    1.095225124053740728e00,
                ],
                [
                    4.788795753668498474e-03,
                    1.001096985136178619e00,
                    -2.276265920899447792e00,
                ],
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.000000000000000000e00,
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

        if output:
            mov_tf = numpy.zeros_like(optical_control_image_data[shift_channel])
            for z in range(mov_tf.shape[0]):
                mov_tf[z, ...] = warp(
                    optical_control_image_data[shift_channel, z, ...],
                    actual_alignment_info.tform,
                    preserve_range=True,
                )

            path = "./camera_alignment_core/tests/tmp_results"
            if not exists(path):
                mkdir(path)
            imsave(
                path + "/test_ref.tiff",
                optical_control_image_data[reference_channel],
            )
            imsave(
                path + "/test_mov.tiff",
                optical_control_image_data[shift_channel],
            )
            imsave(
                path + "/test_mov_tf.tiff",
                mov_tf,
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

        assert abs(matDet - 1) < 0.002

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
