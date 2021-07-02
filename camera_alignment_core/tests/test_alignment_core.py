from camera_alignment_core.alignment_core import AlignmentCore


class TestAlignmentCore:
    def setup_method(self):
        # You can use this to setup before each test
        self.alignment_core = AlignmentCore()

    def test_generate_alignment_matrix(self):
        # Arrange
        optical_control_image = "..."  # TODO
        reference_channel = 0
        shift_channel = 1
        magnification = 100

        expected = []  # TODO

        # Act
        actual = self.alignment_core.generate_alignment_matrix(
            optical_control_image,
            reference_channel,
            shift_channel,
            magnification
        )

        # Assert
        assert actual == expected
