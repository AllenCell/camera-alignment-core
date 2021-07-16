import dataclasses

import numpy.typing


@dataclasses.dataclass
class AlignmentInfo:
    """These are metrics captured/measured as part of generating the alignment matrix."""

    # This represents the rotation of the shifted channel to match the reference channel.
    rotation: int

    # This is blah.
    shift_x: int

    # More good documentation....
    shift_y: int

    # Even better documentation...
    z_offset: int

    # Still better documentation ...
    scaling: float

    # The best documentation ...
    tform: numpy.typing.NDArray
