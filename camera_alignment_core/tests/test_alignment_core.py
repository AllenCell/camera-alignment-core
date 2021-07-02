import numpy

from camera_alignment_core.alignment_core import (
    AlignmentCore,
)


class TestAlignmentCore:
    def setup_method(self):
        # You can use this to setup before each test
        self.alignment_core = AlignmentCore()

    def test_generate_alignment_matrix(self):
        # Arrange
        # TODO, use a test resource that lives in "public-dev-objects.allencell.org" S3 bucket
        optical_control_image = numpy.random.rand(2, 2, 2, 2)
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
            optical_control_image, reference_channel, shift_channel, magnification
        )

        # Assert
        assert actual_alignment_matrix == expected_matrix
        assert actual_alignment_info.shift_y == 2
