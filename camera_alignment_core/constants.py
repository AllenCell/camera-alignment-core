import enum
import typing

LOGGER_NAME = "camera_alignment_core"


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
            # The cropping dimensions for 63x and 20x were agreed upon by the microscopy committee on 9/4/2021:
            #   - Ran 180 argolight field of ring images through the generate_alignment_matrix code
            #   - From the metrics generated from that script, plotted what the crop dimensions would be for each
            #   argolight images such that no black pixels would exist in the image after alignment ("perfect" crop)
            #   - Proceeded to look at how much area was lost due to the above cropping, and plotted this as well
            #   - Matched the area lost from the normal cropping in 100X (~6.3%), and saw how many "perfect" crops
            #   would fall above or below that threshold
            #   - Saw that only 5 of the 180 would wall above the 6.3% area lost when cropped "perfectly"
            #   - Went through these 5 outliers and saw that there were issues with segmentations/SNR/physical
            #   alignment, shwoing that we should fail the outliers.
            #   - Agreed that the dimensions that allow loss of 6.3% of total area (1800x1200) for 20X and 63X
            #   were most suitable for default cropping dimensions
            return CroppingDimension(1800, 1200)
        elif self == Magnification.TWENTY:
            # See notes re 63X magnification for process by which these cropping dimensions were determined
            return CroppingDimension(1800, 1200)

        raise ValueError(f"No cropping dimension defined for {self}")
