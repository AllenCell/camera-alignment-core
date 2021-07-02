# camera-alignment-core

[![Build Status](https://github.com/aics-int/camera_alignment_core/workflows/Build%20Main/badge.svg)](https://github.com/aics-int/camera_alignment_core/actions)
[![Documentation](https://github.com/aics-int/camera_alignment_core/workflows/Documentation/badge.svg)](https://aics-int.github.io/camera_alignment_core/)
[![Code Coverage](https://codecov.io/gh/aics-int/camera_alignment_core/branch/main/graph/badge.svg)](https://codecov.io/gh/aics-int/camera_alignment_core)

Core algorithms for aligning two-camera microscopy imagery

---

## Quick Start

TODO

## Installation

TODO
**Stable Release:** `pip install camera_alignment_core`<br>
**Development Head:** `pip install git+https://github.com/aics-int/camera_alignment_core.git`

## Documentation

For full package documentation please visit [aics-int.github.io/camera_alignment_core](https://aics-int.github.io/camera_alignment_core).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Four Commands You Need To Know

1. `make install`

    This will setup a virtual environment local to this project and install all of the
    project's dependencies into it. The virtual env will be located in `camera-alignment-core/venv`.

2. `make test`

    Run tests. Do this often!

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean workspace.

4. `make docs`

    This will generate the most up-to-date documentation for your Python package.

#### Additional Optional Setup Steps:

-   Turn your project into a GitHub repository:
    -   Make an account on [github.com](https://github.com)
    -   Go to [make a new repository](https://github.com/new)
    -   _Recommendations:_
        -   _It is strongly recommended to make the repository name the same as the Python
            package name_
        -   _A lot of the following optional steps are *free* if the repository is Public,
            plus open source is cool_
    -   After a GitHub repo has been created, run the commands listed under:
        "...or push an existing repository from the command line"
-   Register your project with Codecov:
    -   Make an account on [codecov.io](https://codecov.io)(Recommended to sign in with GitHub)
        everything else will be handled for you.
-   Ensure that you have set GitHub pages to build the `gh-pages` branch by selecting the
    `gh-pages` branch in the dropdown in the "GitHub Pages" section of the repository settings.
    ([Repo Settings](https://github.com/aics-int/camera_alignment_core/settings))
-   Register your project with PyPI:
    -   Make an account on [pypi.org](https://pypi.org)
    -   Go to your GitHub repository's settings and under the
        [Secrets tab](https://github.com/aics-int/camera_alignment_core/settings/secrets/actions),
        add a secret called `PYPI_TOKEN` with your password for your PyPI account.
        Don't worry, no one will see this password because it will be encrypted.
    -   Next time you push to the branch `main` after using `bump2version`, GitHub
        actions will build and deploy your Python package to PyPI.

#### Suggested Git Branch Strategy

1. `main` is for the most up-to-date development, very rarely should you directly
   commit to this branch. GitHub Actions will run on every push and on a CRON to this
   branch but still recommended to commit to your development branches and make pull
   requests to main. If you push a tagged commit with bumpversion, this will also release to PyPI.
2. Your day-to-day work should exist on branches separate from `main`. Even if it is
   just yourself working on the repository, make a PR from your working branch to `main`
   so that you can ensure your commits don't break the development head. GitHub Actions
   will run on every push to any branch or any pull request from any branch to any other
   branch.
3. It is recommended to use "Squash and Merge" commits when committing PR's. It makes
   each set of changes to `main` atomic and as a side effect naturally encourages small
   well defined PR's.


**Allen Institute Software License**

