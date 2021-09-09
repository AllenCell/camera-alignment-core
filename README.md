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


#### CLI Usage

```
align --help
usage: align [-h] --out-dir OUT_DIR --magnification {100,63,20} [--manifest-file MANIFEST_FILE] [--scene SCENE] [--timepoint TIMEPOINT]
             [--ref-channel {405,488,561,638}] [--align-channel {405,488,561,638}] [--no-crop] [-d]
             image optical_control

Run given file through camera alignment, outputting single file per scene.

positional arguments:
  image                 Microscopy image that requires alignment. Passed directly to aicsimageio.AICSImage constructor.
  optical_control       Optical control image to use to align `image`. Passed directly to aicsimageio.AICSImage constructor.

optional arguments:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR     Save output into `out-dir`
  --magnification {100,63,20}
                        Magnification at which both `image` and `optical_control` were acquired.
  --manifest-file MANIFEST_FILE
                        Path to file at which manifest of output of this script will be written. See camera_alignment_core.bin.alignment_output_manifest.
  --scene SCENE         On which scene or scenes within `image` to align. If not specified, will align all scenes within `image`.
  --timepoint TIMEPOINT
                        On which timepoint or timepoints within `image` to perform the alignment. If not specified, will align all timepoints within
                        `image`.
  --ref-channel {405,488,561,638}
                        Which channel of `optical_control` to treat as the 'reference' for alignment. I.e., the 'static' channel. Defined in terms of the
                        wavelength used in that channel.
  --align-channel {405,488,561,638}
                        Which channel of `optical_control` to align, relative to 'reference.' I.e., the 'moving' channel. Defined in terms of the
                        wavelength used in that channel.
  --no-crop             Do not to crop the aligned image(s).
  -d, --debug
```


#### API Documentation

For full package documentation please visit [aics-int.github.io/camera-alignment-core](https://aics-int.github.io/camera-alignment-core/).


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

