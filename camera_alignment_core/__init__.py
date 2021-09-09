from .alignment_core import AlignmentCore

__author__ = "AICS Software"
__email__ = "aditya.nath@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "1.0.0.dev0"


def get_module_version():
    return __version__


__all__ = ("AlignmentCore", "get_module_version")
