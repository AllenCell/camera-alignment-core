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
            # The steps by which these dimensions were determined are unknown--they are "inherited constants."
            return CroppingDimension(900, 600)
        elif self == Magnification.SIXTY_THREE:
            # The cropping dimensions for 63x and 20x were determined by Filip on 8/19/2021 via following steps:
            #   - Look through a sample of ZSD images at each magnification to determine the minimum FOV dimensions. In both cases it was 1848x1248.
            #   - Open a minimum-size image with a brightfield channel in FIJI and duplicate one slice of the bright channel.
            #   - Apply a translation in x and y and a rotation to the image that corresponds to the **assumed worst-case misalignment**:
            #       - Assumed that the microscope user/bead aligner made a decent attempt to visually align the cameras visually
            #       - Applied a translation of **5 pixels in both x and y** and a **rotation of 0.5 degrees**
            #   - Use the line tool to measure the length of the largest regions of black pixels in the resulting image along each axis
            #   - Multiply those numbers by 2 and subtract from the minimum dimensions
            #   Found that 6.3% of image was cropped in 100X (using 924x624 to 900x600). Kept this constant for 63X and 20X to get 1800x1200
            #   Argolight images that were analyzed were shown to have no black pixels once cropped with these dimensions,
            #   other than 5 outliers which would be failed.
            return CroppingDimension(1800, 1200)
        elif self == Magnification.TWENTY:
            # See notes re 63X magnification for process by which these cropping dimensions were determined
            return CroppingDimension(1800, 1200)

        raise ValueError(f"No cropping dimension defined for {self}")
