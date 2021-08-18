"""Test fixtures automatically searched-for and made accessible by pytest."""
import functools
import logging
import pathlib
import shutil
import tempfile
import typing
import urllib.request

from aicsimageio import AICSImage
import pytest

from camera_alignment_core.constants import (
    LOGGER_NAME,
)

log = logging.getLogger(LOGGER_NAME)


def download_file(url: str, to_path: pathlib.Path) -> None:
    log.debug(f"Downloading {url} to {to_path}")
    with urllib.request.urlopen(url) as src, to_path.open(mode="w+b") as dst:
        shutil.copyfileobj(src, dst)


@pytest.fixture
def get_image() -> typing.Callable[[str], typing.Tuple[AICSImage, pathlib.Path]]:
    """Abstraction for returning an AICSImage object from an image path."""
    tmp_dir = (
        pathlib.Path(tempfile.gettempdir()) / "camera-alignment-core-test-fixtures"
    )
    tmp_dir.mkdir(exist_ok=True)

    @functools.lru_cache
    def _get_image(image_uri: str) -> typing.Tuple[AICSImage, pathlib.Path]:
        log.debug("Constructing AICSImage instance for %s", image_uri)

        # Need to download the file if hosted on remote server
        if image_uri.startswith("http"):
            path = pathlib.Path(image_uri)
            target_path = tmp_dir / path.name

            if not target_path.exists():
                download_file(image_uri, target_path)

            return (AICSImage(target_path), target_path)
        else:
            # Assume file system path that is accessible
            return (AICSImage(image_uri), pathlib.Path(image_uri))

    return _get_image
