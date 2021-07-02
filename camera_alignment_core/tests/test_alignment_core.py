import numpy
from aicsimageio import AICSImage

from camera_alignment_core.alignment_core import (
    AlignmentCore,
)


class TestAlignmentCore:
    TEST_RESOURCES_BUCKET = "public-dev-objects.allencell.org"
    ZSD_100x_OPTICAL_CONTROL_IMAGE_KEY = (
        "camera-alignment-core/optical-controls/argo_ZSD1_100X_SLF-015_20210624.czi"
    )

    def setup_method(self):
        # You can use this to setup before each test
        self.zsd_100x_optical_control_image = AICSImage(
            f"s3://{TestAlignmentCore.TEST_RESOURCES_BUCKET}/{TestAlignmentCore.ZSD_100x_OPTICAL_CONTROL_IMAGE_KEY}"
        )
        self.alignment_core = AlignmentCore()

    def test_generate_alignment_matrix(self):
        # Arrange
        optical_control_image_data = self.zsd_100x_optical_control_image.get_image_data(
            "CZYX"
        )
        reference_channel = 0
        shift_channel = 1
        magnification = 100

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

        # Act
        (
            actual_alignment_matrix,
            actual_alignment_info,
        ) = self.alignment_core.generate_alignment_matrix(
            optical_control_image_data, reference_channel, shift_channel, magnification
        )

        # Assert
        assert actual_alignment_matrix == expected_matrix
        assert actual_alignment_info.shift_y == 2
