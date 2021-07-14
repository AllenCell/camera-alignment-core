"""Test fixtures automatically searched-for and made accessible by pytest."""
import functools
import logging
import os
import pathlib

from aicsimageio import AICSImage
import boto3
import pytest

from camera_alignment_core import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


@pytest.fixture(scope="session")
def get_image(tmp_path_factory: pytest.TempPathFactory) -> AICSImage:
    """
    Abstraction for returning an AICSImage object from an image path.
    Handles the complication of
    """
    tmp_dir = tmp_path_factory.mktemp("data")

    @functools.lru_cache
    def _get_image(image: os.PathLike):
        log.debug("Constructing AICSImage instance for %s", image)
        image_path = pathlib.Path(image)
        if str(image_path).endswith((".czi", ".sld")):
            # Need to download the file if on S3
            if str(image_path).startswith("s3"):
                target_path = tmp_dir / image_path.name
                s3_client = boto3.client("s3")
                _, bucket, *key_parts = image_path.parts
                obj_key = "/".join(key_parts)

                log.debug(f"Downloading {obj_key} from {bucket} to {target_path}")
                s3_client.download_file(
                    Bucket=bucket, Key=obj_key, Filename=str(target_path)
                )
                return AICSImage(target_path)
            else:
                # Assume file system path that is accessible
                return AICSImage(image)

        return AICSImage(image)

    return _get_image
