from setuptools import find_packages, setup


requirements = [
    "aicsimageio[czi] ~= 4.4",
    "numpy ~= 1.21",
    "scikit-image ~= 0.18"
]

dev_requirements = [
    # Test
    "black == 21.7b0",
    "flake8 ~= 4.0.1",
    "isort ~= 5.10.1",
    "mypy ~= 0.910",
    "pytest ~= 6.2.5",
    "pytest-raises ~= 0.11",

    # Dev workflow
    "pre-commit ~= 2.17.0",

    # Build
    "build == 0.7.0",

    # Version
    "bump2version ~= 1.0.1",

    # Publish
    "twine ~= 3.7.1",

    # Documentation generation
    "Sphinx ~= 4.4.0",
    "furo == 2022.1.2",  # Third-party theme (https://pradyunsg.me/furo/quickstart/)
    "m2r2 ~= 0.3.2",  # Sphinx extension for parsing README.md as reST and including in Sphinx docs
]

extra_requirements = {
    "dev": dev_requirements,
}


def readme():
    with open("README.md") as readme_file:
        return readme_file.read()


setup(
    author="AICS Software",
    author_email="!AICS_SW@alleninstitute.org",
    description="Core algorithms for aligning two-camera microscopy imagery",
    install_requires=requirements,
    extras_require=extra_requirements,
    license="Allen Institute Software License",
    long_description=readme(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="camera_alignment_core",
    name="camera_alignment_core",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",  # This is driven by aicsimageio constraints
    url="https://github.com/aics-int/camera_alignment_core",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="1.0.0",
    zip_safe=False,
)
