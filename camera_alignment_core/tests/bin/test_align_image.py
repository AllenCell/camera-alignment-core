import contextlib
import pathlib
from unittest.mock import create_autospec, patch

from aicsfiles import FileManagementSystem
from aicsfiles.model import FMSFile
from aicsimageio import AICSImage

from camera_alignment_core.bin import align_image

from .. import (
    ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
    UNALIGNED_ZSD1_IMAGE_URL,
    get_test_image,
)


class TestAlignImageBinScript:
    def test_aligns_image(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )
        microscopy_image, microscopy_image_path = get_test_image(
            UNALIGNED_ZSD1_IMAGE_URL
        )

        expected_aligned_image_path = (
            tmp_path / f"{pathlib.Path(microscopy_image_path).stem}_aligned.ome.tiff"
        )

        with contextlib.ExitStack() as stack:
            # Ensure that when we query FMS for the record of microscopy image to align
            # and its associated optical control, that we return controlled mocks
            fms_find_one_by_id = stack.enter_context(
                patch("aicsfiles.FileManagementSystem.find_one_by_id", autospec=True),
            )
            input_image_fms_record = create_autospec(FMSFile)
            input_image_fms_record.id = "mock_input_fms_file_id"
            input_image_fms_record.path = microscopy_image_path

            control_image_fms_record = create_autospec(FMSFile)
            control_image_fms_record.id = "mock_optical_control_fms_file_id"
            control_image_fms_record.path = optical_control_image_path

            fms_find_one_by_id.side_effect = [
                input_image_fms_record,
                control_image_fms_record,
            ]

            # Ensure that when we upload to FMS, nothing is actually uploaded as part of the test
            fms_upload_file = stack.enter_context(
                patch("aicsfiles.FileManagementSystem.upload_file", autospec=True)
            )

            def mock_upload_file(
                _: FileManagementSystem, save_path: pathlib.Path, **kwargs
            ):
                save_path.link_to(expected_aligned_image_path)
                upload_file = create_autospec(FMSFile)
                upload_file.id = "mock_upload_file_id"
                upload_file.path = expected_aligned_image_path
                return upload_file

            fms_upload_file.side_effect = mock_upload_file

            cli_args = [
                "mock_input_fms_file_id",
                "mock_optical_control_fms_file_id",
                "--magnification",
                "100",
            ]

            # Act
            align_image.main(cli_args)

            # Assert
            assert expected_aligned_image_path.exists()

            aligned_image = AICSImage(expected_aligned_image_path)
            assert len(aligned_image.scenes) == 1
            assert aligned_image.dims.T == microscopy_image.dims.T
            assert aligned_image.dims.C == microscopy_image.dims.C
            assert aligned_image.dims.Z == microscopy_image.dims.Z
