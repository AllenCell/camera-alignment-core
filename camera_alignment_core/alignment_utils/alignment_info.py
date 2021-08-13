import dataclasses

from skimage.transform import SimilarityTransform


@dataclasses.dataclass
class AlignmentInfo:
    """These are metrics captured/measured as part of generating the alignment matrix."""

    # Rotation of image
    rotation: int

    # Rigid Translation
    shift_x: int
    shift_y: int
    z_offset: int

    # image scaling
    scaling: float

    # Complete transform object
    tform: SimilarityTransform
