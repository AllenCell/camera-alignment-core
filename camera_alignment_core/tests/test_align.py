import pathlib
import shutil
import tempfile
import typing

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import numpy
import pytest

from camera_alignment_core import Align
from camera_alignment_core.channel_info import (
    CameraPosition,
    create_channel_info,
)
from camera_alignment_core.constants import (
    Magnification,
)

from . import (
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


class TestAlign:
    def test_default_behavior(self, auto_clean_tmp_dir: pathlib.Path) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )
        microscopy_image, microscopy_image_path = get_test_image(
            UNALIGNED_ZSD1_IMAGE_URL
        )
        channel_info = create_channel_info(microscopy_image_path)
        back_camera_channels = [
            channel.channel_index
            for channel in channel_info.channels
            if channel.camera_position == CameraPosition.BACK
        ]
        align = Align(
            optical_control_image_path,
            Magnification.ONE_HUNDRED,
            out_dir=auto_clean_tmp_dir,
        )

        # Act
        aligned_scenes = align.align_image(
            microscopy_image_path, channels_to_align=back_camera_channels
        )

        # Assert
        assert len(aligned_scenes) == len(microscopy_image.scenes)
        first_scene = AICSImage(aligned_scenes[0].path)
        assert first_scene.dims.T == microscopy_image.dims.T
        assert first_scene.dims.C == microscopy_image.dims.C
        assert first_scene.dims.Z == microscopy_image.dims.Z
        assert first_scene.dims.Y == Magnification.ONE_HUNDRED.cropping_dimension.y
        assert first_scene.dims.X == Magnification.ONE_HUNDRED.cropping_dimension.x

    def test_aligns_image_without_cropping(
        self, auto_clean_tmp_dir: pathlib.Path
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )
        microscopy_image, microscopy_image_path = get_test_image(
            UNALIGNED_ZSD1_IMAGE_URL
        )
        channel_info = create_channel_info(microscopy_image_path)
        back_camera_channels = [
            channel.channel_index
            for channel in channel_info.channels
            if channel.camera_position == CameraPosition.BACK
        ]
        align = Align(
            optical_control_image_path,
            Magnification.ONE_HUNDRED,
            out_dir=auto_clean_tmp_dir,
        )

        # Act
        aligned_scenes = align.align_image(
            microscopy_image_path,
            channels_to_align=back_camera_channels,
            crop_output=False,
        )

        # Assert
        first_scene = AICSImage(aligned_scenes[0].path)
        assert first_scene.dims.T == microscopy_image.dims.T
        assert first_scene.dims.C == microscopy_image.dims.C
        assert first_scene.dims.Z == microscopy_image.dims.Z
        assert first_scene.dims.Y == microscopy_image.dims.Y
        assert first_scene.dims.X == microscopy_image.dims.X

    @pytest.mark.parametrize(
        ["scene_selection_spec", "expected_files"],
        [
            ([1], ["multiscene_Scene-1_aligned.ome.tiff"]),
            (
                [0, 1, 2],
                [
                    "multiscene_Scene-0_aligned.ome.tiff",
                    "multiscene_Scene-1_aligned.ome.tiff",
                    "multiscene_Scene-2_aligned.ome.tiff",
                ],
            ),
            (
                [0, 2],
                [
                    "multiscene_Scene-0_aligned.ome.tiff",
                    "multiscene_Scene-2_aligned.ome.tiff",
                ],
            ),
            pytest.param([0, 3], [], marks=pytest.mark.xfail(raises=IndexError)),
        ],
    )
    def test_aligns_selected_scenes(
        self,
        scene_selection_spec: typing.List[int],
        expected_files: typing.List[str],
        auto_clean_tmp_dir: pathlib.Path,
        multi_scene_image: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )
        align = Align(
            optical_control_image_path,
            Magnification.ONE_HUNDRED,
            out_dir=auto_clean_tmp_dir,
        )

        # Act
        aligned_scenes = align.align_image(
            multi_scene_image, channels_to_align=[0, 2], scenes=scene_selection_spec
        )

        # Assert
        assert len(aligned_scenes) == len(expected_files)
        for file in aligned_scenes:
            assert file.scene in scene_selection_spec
            assert file.path.name in expected_files

    @pytest.mark.parametrize(
        ["timepoint_selection_spec"],
        [
            ([1],),
            ([0, 1, 2],),
            ([0, 2],),
            pytest.param([0, 3], marks=pytest.mark.xfail(raises=IndexError)),
        ],
    )
    def test_aligns_selected_timepoints(
        self,
        timepoint_selection_spec: typing.List[int],
        auto_clean_tmp_dir: pathlib.Path,
        multi_timepoint_image: pathlib.Path,
    ) -> None:
        # Arrange
        _, optical_control_image_path = get_test_image(
            ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL
        )

        align = Align(
            optical_control_image_path,
            Magnification.ONE_HUNDRED,
            out_dir=auto_clean_tmp_dir,
        )

        # Act
        aligned_scenes = align.align_image(
            multi_timepoint_image,
            channels_to_align=[0, 2],
            timepoints=timepoint_selection_spec,
        )

        # Assert
        assert len(aligned_scenes) == 1
        aligned_image_info = aligned_scenes[0]
        assert AICSImage(aligned_image_info.path).dims.T == len(
            timepoint_selection_spec
        )
