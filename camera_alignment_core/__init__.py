from .align import Align
from .constants import Magnification

__author__ = "AICS"

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "1.0.6"


def get_module_version():
    return __version__


__all__ = ("Align", "get_module_version", "Magnification")
