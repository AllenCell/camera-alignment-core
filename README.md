# camera-alignment-core


Core algorithms for aligning two-camera microscopy imagery

---


## Installation

**Stable Release:** `pip install camera_alignment_core`<br>

This library is published to a private PyPI server ("Artifactory") accessible within the Allen Institute network or over VPN. This has downstream effects for how this library is installed into other Python packages.

Having trouble installing? Start here: http://confluence.corp.alleninstitute.org/display/SF/Using+Artifactory#UsingArtifactory-Python.


## Documentation

The primary export of this package is the [Align](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.align.Align) class.
It provides a convenient abstraction over what is expected to be the most common usage of this package. Example use:
```python
from camera_alignment_core import Align, Channel, Magnification


align = Align(
    optical_control="/some/path/to/an/argolight-field-of-rings.czi",
    magnification=Magnification(20),
    reference_channel=Channel.RAW_561_NM,
    alignment_channel=Channel.RAW_638_NM,
    out_dir="/tmp/whereever",
)
aligned_scenes = align.align_image("/some/path/to/an/image.czi")
aligned_optical_control = align.align_optical_control()
alignment_matrix = align.alignment_transform.matrix
alignment_info = align.alignment_transform.info
```

##### Low-level API
In addition, the lower-level functional building blocks used internally by [Align](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.align.Align) are accessible in the `camera_alignment_core.alignment_core` module. See:
1. [align_image](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.align_image)
1. [apply_alignment_matrix](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.apply_alignment_matrix)
1. [crop](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.crop)
1. [generate_alignment_matrix](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.generate_alignment_matrix)
1. [get_channel_info](https://aics-int.github.io/camera-alignment-core/camera_alignment_core.html#camera_alignment_core.alignment_core.get_channel_info)


## Development

1. `make install`

    This will setup a virtual environment local to this project and install all of the
    project's dependencies into it. The virtual env will be located in `camera-alignment-core/venv`.

2. `make test`, `make fmt`, `make lint`, `make type-check`, `make import-sort`

    Quality assurance

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean workspace.


**Allen Institute Software License**

