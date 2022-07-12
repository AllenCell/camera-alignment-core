# camera-alignment-core


Core algorithms for aligning two-camera microscopy imagery

---


## Installation

`pip install camera_alignment_core==1.0.3`<br>

This library is published to a private PyPI server ("Artifactory") accessible within the Allen Institute network or over VPN. This has downstream effects for how this library is installed into other Python packages.

Having trouble installing? Start here: http://confluence.corp.alleninstitute.org/display/SF/Using+Artifactory#UsingArtifactory-Python.


## Documentation

The primary export of this package is the [Align](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.align.Align) class.
It provides a convenient abstraction over what is expected to be the most common usage of this package. Example use:
```python
from camera_alignment_core import Align, Magnification
from camera_alignment_core.channel_info import channel_info_factory, CameraPosition


biological_data_to_align_path = "/some/path/to/an/image.czi"
optical_control_path = "/some/path/to/an/argolight-field-of-rings.czi"

# `Align` is the root, coordinating entity, that encapsulates all of the steps to:
#   1. Generate a similarity transformation matrix;
#   2. Apply that matrix to a (potentially) multi-scene, (potentially) multi-timepoint biological image;
#   3. Apply that matrix to the optical_control itself
#       (e.g., to produce a reference alignment that can be consulted to assess alignment quality).
#
# This snippet shows using only the required parameters, but `Align` also takes optional keyword arguments
# for specifying which channels within `optical_control` to treat as the
# reference and the shift for the purpose of producing the similarity transformation matrix.
# See `Align`'s documentation for details on the default behavior if these optional arguments
# are not provided.
align = Align(
    optical_control=optical_control_path,
    magnification=Magnification(20),
    out_dir="/tmp/whereever",
)

# If you don't already have a list of channel indices that you know should be shifted
# during alignment, consider using the `ChannelInfo` utility. Note:
#   1. You must construct a `ChannelInfo` using the `channel_info_factory`.
#   2. Currently, only CZI files are supported by this utility.
biological_data_channel_info = channel_info_factory(biological_data_to_align_path)

# ...`ChannelInfo` offers convenience methods for identifying an image's channels.
# For example, identifying which were acquired on the back camera (e.g.: Brightfield, CMDRP).
biological_data_back_channels = biological_data_channel_info.channels_from_camera_position(
    CameraPosition.BACK
)

# `Align::align_image` returns a list of info objects (`AlignedImage`) pointing at the output of the method.
# These info objects have two properties:
#   1. scene : the scene index from the image that this newly aligned image is from; and
#   2. path : the filesystem path to the aligned scene. The base path is the `out_dir` you
#       specified in the `Align` constructor.
# If the image you're aligning is single-scene, this will be a list of one `AlignedImage`.
aligned_scenes = align.align_image(
    image_to_align_path,
    channels_to_shift=[channel.channel_index for channel in biological_data_back_channels]
)

# ...(Right now in the script would be a great time to iterate over `aligned_scenes` and
# upload each aligned scene to FMS and delete the now unnecessary versions saved in `out_dir`.
# Refer to aicsfiles documentation should you need help with uploading.)...

# You can also use `Align` to create a reference alignment by aligning the optical control
# image itself. This may be helpful, for example, to spot check alignnment quality.
optical_control_channel_info = channel_info_factory(optical_control_path)
optical_control_back_channels = optical_control_channel_info.channels_from_camera_position(
    CameraPosition.BACK
)
aligned_optical_control = align.align_optical_control(
    channels_to_shift=[channel.channel_index for channel in optical_control_back_channels]
)

# `Align` also provides access to certain details from the alignment process,
# including the similarity transformation matrix itself, as well as summary information
# gathered while producing that matrix (see documentation for `AlignmentInfo` for more details).
alignment_matrix = align.alignment_transform.matrix
alignment_info = align.alignment_transform.info
```

##### Low-level API
In addition, the lower-level functional building blocks used internally by [Align](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.align.Align) are accessible in the `camera_alignment_core.alignment_core` module. See:
1. [align_image](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.align_image)
1. [apply_alignment_matrix](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.apply_alignment_matrix)
1. [crop](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.crop)
1. [generate_alignment_matrix](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.generate_alignment_matrix)


## Development
This repository uses `make` as a task runner. Various `make` commands/targets have been written to automate
the setup of a local development environment, run quality assurance tests, build and distribute the
library. The following are notable `make` targets;
see `Makefile` for others or to inspect the underlying scripts run as part of these targets:

1. `make install`:

    This will setup a virtual environment local to this project and install all of the
    project's dependencies into it. The virtual env will be located in `camera-alignment-core/venv`.

2. Quality assurance `make` tasks:

    After running `make install`, a `git` pre-commit hook will be installed in your local `.git/hooks`.
    This pre-commit hook is managed by `pre-commit`, configured in `.pre-commit-config.yaml`.
    These commit hooks will ensure QA is run on your code before you commit it.
    The code formatters that are run on pre-commit have side-effects;
    if they don't exit successfully, they will modify your staged files, and you will need to stage the new changes.

    1. `make lint`: Linting using `flake8`
    2. `make type-check`: Static analysis using `mypy`
    3. `make fmt`: Auto-code formatting using `black`
    4. `make import-sort`: Auto-import sorting using `isort`
    5. `make test`: Programmatic tests, run with `pytest`

    Code is not ready to merge until all of these QA tests pass.

3. `make clean`:

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean workspace. This, in combination with `make install`,
    should be the first thing you do if your branch is building locally but failing in CI.


**Allen Institute Software License**

