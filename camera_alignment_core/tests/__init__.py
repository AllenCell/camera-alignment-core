import logging
import pathlib
import shutil
import tempfile
import typing
import urllib.request

from aicsimageio import AICSImage

from camera_alignment_core.constants import (
    LOGGER_NAME,
)

log = logging.getLogger(LOGGER_NAME)

# --- TEST RESOURCES ---
# Taken from /allen/aics/microscopy/PRODUCTION/OpticalControl/ArgoLight/Argo_QC_Daily/ZSD1/ZSD1_argo_100X_SLF-015_20210624
ZSD_100x_OPTICAL_CONTROL_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_ZSD1_100X_SLF-015_20210624.czi"

# FMS ID: 0023c446cd384dc3947c90dc7a76f794; 303.38 MB
GENERIC_OME_TIFF_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/3500003897_100X_20200306_1r-Scene-30-P89-G11.ome.tiff"

# FMS ID: 439c852ea76e46d4b9a9f8813f331b4d; 264.43 MB
GENERIC_CZI_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/20200701_N04_001.czi"

UNALIGNED_ZSD1_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/images/3500004473_100X_20210430_1c-Scene-24-P96-G06.czi"
ALIGNED_ZSD1_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/aligned-images/3500004473_100X_20210430_1c-Scene-24-P96-G06_aligned.ome.tiff"
ARGOLIGHT_OPTICAL_CONTROL_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_100X_20210430_fieldofrings.czi"
ARGOLIGHT_ALIGNED_IMAGE_URL = "https://s3.us-west-2.amazonaws.com/public-dev-objects.allencell.org/camera-alignment-core/optical-controls/argo_100X_20210430_fieldofrings_aligned.tif"


TEST_RESOURCE_DOWNLOAD_DIRECTORY = (
    pathlib.Path(tempfile.gettempdir()) / "camera-alignment-core-test-fixtures"
)


def get_test_image(
    image_uri: str, resource_directory: pathlib.Path = TEST_RESOURCE_DOWNLOAD_DIRECTORY
) -> typing.Tuple[AICSImage, pathlib.Path]:
    """
    Abstraction for returning an AICSImage object from an image path.
    Will
    """
    resource_directory.mkdir(exist_ok=True)

    log.debug("Constructing AICSImage instance for %s", image_uri)

    # Need to download the file if hosted on remote server
    if image_uri.startswith("http"):
        path = pathlib.Path(image_uri)
        target_path = resource_directory / path.name

        if not target_path.exists():
            log.debug("Downloading %s to %s", image_uri, target_path)
            with urllib.request.urlopen(image_uri) as src, target_path.open(
                mode="w+b"
            ) as dst:
                shutil.copyfileobj(src, dst)

        return (AICSImage(target_path), target_path)
    else:
        # Assume file system path that is accessible
        return (AICSImage(image_uri), pathlib.Path(image_uri))
