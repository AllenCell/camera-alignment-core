"""Test fixtures automatically searched-for and made accessible by pytest."""
import functools
import logging
import pathlib
import shutil
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


@pytest.fixture(scope="session")
def get_image(tmp_path_factory: pytest.TempPathFactory) -> AICSImage:
    """
    Abstraction for returning an AICSImage object from an image path.
    Handles the complication of
    """
    tmp_dir = tmp_path_factory.mktemp("data")

    @functools.lru_cache
    def _get_image(image: str):
        log.debug("Constructing AICSImage instance for %s", image)

        # Need to download the file if hosted on remote server
        if image.startswith("http"):
            path = pathlib.Path(image)
            target_path = tmp_dir / path.name
            download_file(image, target_path)
            return AICSImage(target_path)
        else:
            # Assume file system path that is accessible
            return AICSImage(image)

    return _get_image
