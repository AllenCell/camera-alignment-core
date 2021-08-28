import pathlib
import shutil
import tempfile
import typing

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import numpy
import pytest

from camera_alignment_core.bin import (
    AlignmentOutputManifest,
    align_image,
)
from camera_alignment_core.bin.image_dimension_action import (
    parse_dimension_str,
)
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

        output_manifest_path = auto_clean_tmp_dir / "manifest.json"

        cli_args = [
            str(microscopy_image_path),
            str(optical_control_image_path),
            "--out-dir",
            str(auto_clean_tmp_dir),
            "--magnification",
            str(Magnification.ONE_HUNDRED.value),
            "--manifest-file",
            str(output_manifest_path),
        ]

        # Act
        align_image.main(cli_args)
        output = AlignmentOutputManifest.from_file(output_manifest_path)

        # Assert
        assert len(output.aligned_images) == 1
        aligned_image_info = output.aligned_images[0]
        assert aligned_image_info.path.exists()

        aligned_image = AICSImage(aligned_image_info.path)
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

        output_manifest_path = auto_clean_tmp_dir / "manifest.json"

        cli_args = [
            str(multi_scene_image),
            str(optical_control_image_path),
            "--out-dir",
            str(auto_clean_tmp_dir),
            "--magnification",
            str(Magnification.ONE_HUNDRED.value),
            "--manifest-file",
            str(output_manifest_path),
            "--scene",
            scene_selection_spec,
        ]

        expected_scenes = parse_dimension_str(scene_selection_spec)

        # Act
        align_image.main(cli_args)
        output = AlignmentOutputManifest.from_file(output_manifest_path)

        # Assert
        assert len(output.aligned_images) == len(expected_files)
        for file in output.aligned_images:
            assert file.scene in expected_scenes
            assert file.path.name in expected_files

    @pytest.mark.parametrize(
        ["timepoint_selection_spec"],
        [
            ("1",),
            ("0-2",),
            ("0, 2",),
            pytest.param("0, 3", marks=pytest.mark.xfail(raises=IndexError)),
        ],
    )
    def test_aligns_selected_timepoints(
        self,
        timepoint_selection_spec: str,
        auto_clean_tmp_dir: pathlib.Path,
        multi_timepoint_image: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )

        output_manifest_path = auto_clean_tmp_dir / "manifest.json"

        cli_args = [
            str(multi_timepoint_image),
            str(optical_control_image_path),
            "--out-dir",
            str(auto_clean_tmp_dir),
            "--magnification",
            str(Magnification.ONE_HUNDRED.value),
            "--manifest-file",
            str(output_manifest_path),
            "--timepoint",
            timepoint_selection_spec,
        ]

        expected_timepoints = parse_dimension_str(timepoint_selection_spec)

        # Act
        align_image.main(cli_args)
        output = AlignmentOutputManifest.from_file(output_manifest_path)

        # Assert
        assert len(output.aligned_images) == 1
        aligned_image_info = output.aligned_images[0]
        assert AICSImage(aligned_image_info.path).dims.T == len(expected_timepoints)
