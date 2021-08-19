import enum
import typing

LOGGER_NAME = "camera_alignment_core"


class Channel(enum.Enum):
    """Standardized channel names"""

    RAW_BRIGHTFIELD = "Raw brightfield"
    RAW_405_NM = "Raw 405nm"
    RAW_488_NM = "Raw 488nm"
    RAW_561_NM = "Raw 561nm"
    RAW_638_NM = "Raw 638nm"

    def requires_alignment(self) -> bool:
        if self in (Channel.RAW_BRIGHTFIELD, Channel.RAW_638_NM):
            return True
        return False


class CroppingDimension(typing.NamedTuple):
    x: int
    y: int


class Magnification(enum.Enum):
    """Supported magnification values."""

    ONE_HUNDRED = 100
    SIXTY_THREE = 63
    TWENTY = 20

    @property
    def cropping_dimension(self) -> CroppingDimension:
        if self == Magnification.ONE_HUNDRED:
            return CroppingDimension(900, 600)
        elif self == Magnification.SIXTY_THREE:
            # This is a placeholder
            return CroppingDimension(1200, 900)
        elif self == Magnification.TWENTY:
            # This is a placeholder
            return CroppingDimension(1600, 1200)

        raise ValueError(f"No cropping dimension defined for {self}")
