#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "black ~= 21.6b0",
    "flake8 ~= 3.9",
    "isort ~= 5.9",
    "mypy ~= 0.910",
    "pre-commit ~= 2.13",
    "pytest-runner ~= 5.2",
]

test_requirements = [
    "codecov ~= 2.1.4",
    "pytest ~= 6.2",
    "pytest-cov ~= 2.12",
    "pytest-raises ~= 0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version ~= 1.0",
    "coverage ~= 5.1",
    "m2r2 ~= 0.2",
    "pytest-runner ~= 5.3",
    "Sphinx ~= 4.0",
    "sphinx_rtd_theme ~= 0.5",
    "tox ~= 3.15",
    "twine ~= 3.4",
    "wheel ~= 0.36",
]

requirements = [
    "aicsimageio[czi] == 4.0.3",
    "numpy ~= 1.21",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ]
}

setup(
    author="AICS Software",
    author_email="aditya.nath@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Core algorithms for aligning two-camera microscopy imagery",
    entry_points={
        "console_scripts": [
            "my_example=camera_alignment_core.bin.my_example:main"
        ],
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="camera_alignment_core",
    name="camera_alignment_core",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",  # This is driven by aicsimageio constraints
    setup_requires=setup_requirements,
    test_suite="camera_alignment_core/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/aics-int/camera_alignment_core",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
