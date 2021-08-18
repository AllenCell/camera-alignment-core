import logging

import numpy
import numpy.testing
import numpy.typing
import pytest

from camera_alignment_core import AlignmentCore
from camera_alignment_core.channel_info import (
    ChannelInfo,
)
from camera_alignment_core.constants import (
    LOGGER_NAME,
    Channel,
    Magnification,
)
from camera_alignment_core.exception import (
    IncompatibleImageException,
    UnsupportedMagnification,
)

from . import (
    ALIGNED_ZSD1_IMAGE_URL,
    ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
    GENERIC_CZI_URL,
    GENERIC_OME_TIFF_URL,
    UNALIGNED_ZSD1_IMAGE_URL,
    ZSD_100x_OPTICAL_CONTROL_IMAGE_URL,
    get_test_image,
)

log = logging.getLogger(LOGGER_NAME)


class TestAlignmentCore:
    def setup_method(self):
        """You can use this to setup before each test"""
        self.alignment_core = AlignmentCore()

    @pytest.mark.slow
    def test_generate_alignment_matrix(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        optical_control_image, _ = get_test_image(ZSD_100x_OPTICAL_CONTROL_IMAGE_URL)
        control_image_channel_map = self.alignment_core.get_channel_name_to_index_map(
            optical_control_image
        )
        optical_control_image_data = optical_control_image.get_image_data("CZYX", T=0)

        # Transform matrix for this same test resource produced by Calysta's version of this code found here:
        # /allen/aics/microscopy/PRODUCTION/OpticalControl/ArgoLight/Argo_QC_Daily/ZSD1/ZSD1_argo_100X_SLF-015_20210624/argo_ZSD1_100X_SLF-015_20210624_sim_matrix.txt
        # [
        #     [
        #         1.001828593258253797e00,
        #         -5.167305751106508228e-03,
        #         -5.272046139691610733e-02,
        #     ],
        #     [
        #         5.167305751106508228e-03,
        #         1.001828593258254241e00,
        #         -3.061419473755620402e00,
        #     ],
        #     [
        #         0.000000000000000000e00,
        #         0.000000000000000000e00,
        #         1.000000000000000000e00,
        #     ],
        # ]

        expected_matrix = numpy.array(
            [
                [1.00142517, -0.00520853, -0.05273507],
                [0.00520853, 1.00142517, -3.17384935],
                [0.0, 0.0, 1.0],
            ]
        )

        # Act
        (
            actual_alignment_matrix,
            actual_alignment_info,
        ) = self.alignment_core.generate_alignment_matrix(
            optical_control_image_data,
            reference_channel=control_image_channel_map["Raw 405nm"],
            shift_channel=control_image_channel_map["Raw 638nm"],
            magnification=100,
            px_size_xy=optical_control_image.physical_pixel_sizes.X,
        )

        # Assert
        assert actual_alignment_matrix.shape == expected_matrix.shape
        assert numpy.allclose(actual_alignment_matrix, expected_matrix), (
            actual_alignment_matrix - expected_matrix
        )

    @pytest.mark.slow
    def test_generate_alignment_matrix_reproducability(
        self,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        optical_control_image, _ = get_test_image(ZSD_100x_OPTICAL_CONTROL_IMAGE_URL)
        optical_control_image_data = optical_control_image.get_image_data("CZYX")
        control_image_channel_map = self.alignment_core.get_channel_name_to_index_map(
            optical_control_image
        )
        reference_channel = control_image_channel_map["Raw 405nm"]
        shift_channel = control_image_channel_map["Raw 638nm"]
        magnification = 100
        pixel_size_xy = optical_control_image.physical_pixel_sizes.X

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

        assert numpy.array_equal(alignment_matrix_1, alignment_matrix_2)

    @pytest.mark.parametrize(
        ["image_path", "expectation"],
        [
            (
                GENERIC_OME_TIFF_URL,
                ChannelInfo(
                    {
                        Channel.RAW_BRIGHTFIELD: 0,
                        Channel.RAW_488_NM: 1,
                        Channel.RAW_638_NM: 2,
                        Channel.RAW_405_NM: 3,
                    }
                ),
            ),
            (GENERIC_CZI_URL, ChannelInfo({Channel.RAW_561_NM: 1})),
        ],
    )
    @pytest.mark.slow
    def test_get_channel_info(
        self,
        image_path,
        expectation,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        image, _ = get_test_image(image_path)

        # Act
        result = self.alignment_core.get_channel_info(image)

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
        caplog: pytest.LogCaptureFixture,
    ):

        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        image, _ = get_test_image(image_path)
        optical_control_image, _ = get_test_image(alignment_image_path)
        optical_control_image_data = optical_control_image.get_image_data("CZYX", T=0)
        optical_control_channel_info = self.alignment_core.get_channel_info(
            optical_control_image
        )
        (
            alignment_matrix,
            alignment_info,
        ) = self.alignment_core.generate_alignment_matrix(
            optical_control_image=optical_control_image_data,
            shift_channel=optical_control_channel_info.index_of_channel(
                Channel.RAW_638_NM
            ),
            reference_channel=optical_control_channel_info.index_of_channel(
                Channel.RAW_405_NM
            ),
            magnification=magnification,
            px_size_xy=optical_control_image.physical_pixel_sizes.X,
        )

        image_channel_info = self.alignment_core.get_channel_info(image)

        expectation_image, _ = get_test_image(expectation_image_path)

        # Act
        result = self.alignment_core.align_image(
            alignment_matrix=alignment_matrix,
            image=image.get_image_data("CZYX", T=0),
            channel_info=image_channel_info,
            magnification=magnification,
        )

        # Assert
        assert numpy.array_equal(result, expectation_image.get_image_data("CZYX", T=0))

    @pytest.mark.parametrize(
        [
            "image",
            "alignment_matrix",
            "channel_info",
            "magnification",
            "expected_exception",
        ],
        [
            (
                numpy.random.rand(1, 1, 1, 1, 1),  # Wrong dimensions
                numpy.eye(3, 3),
                ChannelInfo({Channel.RAW_BRIGHTFIELD: 0}),
                100,
                IncompatibleImageException,
            ),
            (
                numpy.random.rand(1, 1, 1, 1),
                numpy.eye(3, 3),
                ChannelInfo({}),  # Empty ChannelInfo
                100,
                ValueError,
            ),
            (
                numpy.random.rand(1, 1, 1, 1),
                numpy.eye(3, 3),
                # No alignable channels
                ChannelInfo(
                    {
                        Channel.RAW_405_NM: 0,
                        Channel.RAW_488_NM: 1,
                        Channel.RAW_561_NM: 2,
                    }
                ),
                100,
                IncompatibleImageException,
            ),
            (
                numpy.random.rand(1, 1, 1, 1),
                numpy.eye(3, 3),
                ChannelInfo({Channel.RAW_BRIGHTFIELD: 0}),
                33,  # Unsupported magnification
                UnsupportedMagnification,
            ),
        ],
    )
    def test_align_image_guards_against_unsupported_parameters(
        self,
        image: numpy.typing.NDArray[numpy.uint16],
        alignment_matrix: numpy.typing.NDArray[numpy.float16],
        channel_info: ChannelInfo,
        magnification: int,
        expected_exception: typing.Type[Exception],
    ):
        # Act / Assert
        with pytest.raises(expected_exception):
            self.alignment_core.align_image(
                alignment_matrix, image, channel_info, magnification
            )

    # TODO: Add 63x and 20x images to test
    @pytest.mark.parametrize(
        ["image_path", "magnification", "expected_shape"],
        [
            (
                UNALIGNED_ZSD1_IMAGE_URL,
                Magnification(100),
                (4, 75, 600, 900),
            ),
        ],
    )
    def test_crop(
        self,
        image_path,
        magnification,
        expected_shape,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
        image, _ = get_test_image(image_path)
        image_data = image.get_image_data("CZYX", T=0)

        # Act
        result = self.alignment_core._crop(image_data, magnification)

        # Assert
        assert result.shape == expected_shape
