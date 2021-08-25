import contextlib
import pathlib
import shutil
import tempfile
import typing
from unittest.mock import create_autospec, patch

from aicsfiles import FileManagementSystem
from aicsfiles.model import FMSFile
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import numpy
import pytest

from camera_alignment_core.bin import align_image
from camera_alignment_core.constants import (
    Magnification,
)

from .. import (
    ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL,
    UNALIGNED_ZSD1_IMAGE_URL,
    get_test_image,
)


@pytest.fixture
def auto_clean_tmp_dir(
    tmp_path: pathlib.Path,
) -> typing.Generator[pathlib.Path, None, None]:
    """
    The built-in tmp_path pytest fixture will be left in place for three test sessions.
    This instead immediately removes the tmp dir.
    """
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="session")
def multi_scene_image() -> typing.Generator[pathlib.Path, None, None]:
    print("Generating multi_scene_image")
    microscopy_image, _ = get_test_image(UNALIGNED_ZSD1_IMAGE_URL)
    scene_image_data = microscopy_image.get_image_data("TCZYX")
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    multi_scene_image_path = tmpdir / "multiscene.ome.tiff"
    num_scenes = 3
    OmeTiffWriter.save(
        channel_names=microscopy_image.channel_names,
        data=[scene_image_data for _ in range(num_scenes)],
        dim_order="TCZYX",
        uri=multi_scene_image_path,
    )

    yield multi_scene_image_path

    shutil.rmtree(tmpdir)


@pytest.fixture(scope="session")
def multi_timepoint_image() -> typing.Generator[pathlib.Path, None, None]:
    print("Generating multi_timepoint_image")
    microscopy_image, _ = get_test_image(UNALIGNED_ZSD1_IMAGE_URL)
    timepoint_image_data = microscopy_image.get_image_data("CZYX", T=0)
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    out_path = tmpdir / "multitimepoint.ome.tiff"
    OmeTiffWriter.save(
        channel_names=microscopy_image.channel_names,
        data=numpy.stack(
            [timepoint_image_data, timepoint_image_data, timepoint_image_data]
        ),
        dim_order="TCZYX",
        uri=out_path,
    )

    yield out_path

    shutil.rmtree(tmpdir)


class TestAlignImageBinScript:
    def test_aligns_image(self, auto_clean_tmp_dir: pathlib.Path) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )
        microscopy_image, microscopy_image_path = get_test_image(
            UNALIGNED_ZSD1_IMAGE_URL
        )

        expected_aligned_image_path = (
            auto_clean_tmp_dir / f"{microscopy_image_path.stem}_aligned.ome.tiff"
        )

        cli_args = [
            str(microscopy_image_path),
            str(optical_control_image_path),
            "--magnification",
            str(Magnification.ONE_HUNDRED.value),
            "--out-dir",
            str(auto_clean_tmp_dir),
        ]

        # Act
        align_image.main(cli_args)

        # Assert
        assert expected_aligned_image_path.exists()

        aligned_image = AICSImage(expected_aligned_image_path)
        assert len(aligned_image.scenes) == len(microscopy_image.scenes)
        assert aligned_image.dims.T == microscopy_image.dims.T
        assert aligned_image.dims.C == microscopy_image.dims.C
        assert aligned_image.dims.Z == microscopy_image.dims.Z
        assert aligned_image.dims.Y == Magnification.ONE_HUNDRED.cropping_dimension.y
        assert aligned_image.dims.X == Magnification.ONE_HUNDRED.cropping_dimension.x

    @pytest.mark.parametrize(
        ["scene_selection_spec", "expected_files"],
        [
            ("1", ["multiscene_Scene-1_aligned.ome.tiff"]),
            (
                "0-2",
                [
                    "multiscene_Scene-0_aligned.ome.tiff",
                    "multiscene_Scene-1_aligned.ome.tiff",
                    "multiscene_Scene-2_aligned.ome.tiff",
                ],
            ),
            (
                "0, 2",
                [
                    "multiscene_Scene-0_aligned.ome.tiff",
                    "multiscene_Scene-2_aligned.ome.tiff",
                ],
            ),
            pytest.param("0, 3", [], marks=pytest.mark.xfail(raises=IndexError)),
        ],
    )
    def test_aligns_selected_scenes(
        self,
        scene_selection_spec: str,
        expected_files: typing.List[str],
        auto_clean_tmp_dir: pathlib.Path,
        multi_scene_image: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )

        cli_args = [
            str(multi_scene_image),
            str(optical_control_image_path),
            "--magnification",
            str(Magnification.ONE_HUNDRED.value),
            "--scene",
            scene_selection_spec,
            "--out-dir",
            str(auto_clean_tmp_dir),
        ]

        # Act
        align_image.main(cli_args)

        # Assert
        matching = sorted(auto_clean_tmp_dir.glob("*_aligned.ome.tiff"))
        assert len(matching) == len(expected_files)
        for file in matching:
            assert file.name in expected_files

    @pytest.mark.parametrize(
        ["timepoint_selection_spec", "expected_timepoints"],
        [
            ("1", 1),
            ("0-2", 3),
            ("0, 2", 2),
            pytest.param("0, 3", None, marks=pytest.mark.xfail(raises=IndexError)),
        ],
    )
    def test_aligns_selected_timepoints(
        self,
        timepoint_selection_spec: str,
        expected_timepoints: int,
        auto_clean_tmp_dir: pathlib.Path,
        multi_timepoint_image: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )

        expected_out_file = auto_clean_tmp_dir / "multitimepoint_aligned.ome.tiff"

        cli_args = [
            str(multi_timepoint_image),
            str(optical_control_image_path),
            "--magnification",
            str(Magnification.ONE_HUNDRED.value),
            "--timepoint",
            timepoint_selection_spec,
            "--out-dir",
            str(auto_clean_tmp_dir),
        ]

        # Act
        align_image.main(cli_args)

        # Assert
        assert expected_out_file.exists()
        assert AICSImage(expected_out_file).dims.T == expected_timepoints

    def test_aligns_image_fms_integation(
        self,
        auto_clean_tmp_dir: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )
        _, microscopy_image_path = get_test_image(UNALIGNED_ZSD1_IMAGE_URL)

        expected_aligned_image_path = (
            auto_clean_tmp_dir
            / f"{pathlib.Path(microscopy_image_path).stem}_aligned.ome.tiff"
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
                str(Magnification.ONE_HUNDRED.value),
            ]

            # Act
            align_image.main(cli_args)

            # Assert
            assert fms_find_one_by_id.call_count == 2
            assert fms_upload_file.called
