# camera-alignment-core


Core algorithms for aligning two-camera microscopy imagery

---


## Installation

**Stable Release:** `pip install camera_alignment_core`<br>

This library is published to a private PyPI server ("Artifactory") accessible within the Allen Institute network or over VPN. This has downstream effects for how this library is installed into other Python packages.

These instructions assume the use of a virtual environment.

---
**NOTE**

It is highly recommended to immediately upgrade `pip` to its latest version after creating a new virtual environment. For example:
```bash
# Create new virtual environment
python3.9 -m venv venv

# Upgrade pip
venv/bin/python3 -m pip install -U pip
```
---

#### Configure `pip`
The recommended way to configure `pip` to install `camera_alignment_core` from our private PyPI server is to add a few directives to your `pip` configuration (`pip.conf`):
```
[global]
index-url = https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple
extra-index-url = https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-snapshot-local/simple

[install]
trusted-host = artifactory.corp.alleninstitute.org
```

`pip` will search for and merge configuration files from a number of locations: system-, user-, and virtual-environment-level configuration files. See [pip](https://pip.pypa.io/en/stable/user_guide/#configuration) documentation for where these files should be created--if not already--depending on your operating system.

You may add this configuration to the `pip.conf` of your choosing, but note that `pip` reads and merges configuration in order, allowing virtual-env-level config to override user-level configuration, which overrides system-level config.

#### Alternate approach: use a requirements.txt file in your repository
Assuming you make use of `setuptools` to manage your Python package's build, you can use `pip` to install the dependencies you've declared in `setup.[py|cfg]`, and in the process, instruct `pip` how to install from our private PyPI server.

This can be done either by using `pip` configuration files (detailed above), or by adding a `requirements.txt` file to your repository. That `requirements.txt` file should contain commandline `pip` options that accomplish the same as the directives detailed in the section above, and then instruct `pip` to install packages from `setup.[py|cfg]`. An example `requirements.txt` file:
```
--trusted-host artifactory.corp.alleninstitute.org
--index-url https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple
--extra-index-url https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-snapshot-local/simple

-e .[all]
```

## Documentation

The primary export of this package is the [Align](/camera_alignment_core.html#camera_alignment_core.align.Align) class.
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
In addition, the lower-level functional building blocks used internally by [Align](/camera_alignment_core.html#camera_alignment_core.align.Align) are accessible in the `camera_alignment_core.alignment_core` module. See:
1. [align_image](/camera_alignment_core.html#camera_alignment_core.alignment_core.align_image)
1. [apply_alignment_matrix](/camera_alignment_core.html#camera_alignment_core.alignment_core.apply_alignment_matrix)
1. [crop](/camera_alignment_core.html#camera_alignment_core.alignment_core.crop)
1. [generate_alignment_matrix](/camera_alignment_core.html#camera_alignment_core.alignment_core.generate_alignment_matrix)
1. [get_channel_info](/camera_alignment_core.html#camera_alignment_core.alignment_core.get_channel_info)


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

