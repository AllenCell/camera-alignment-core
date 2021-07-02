import typing

import numpy.typing

from .alignment_info import AlignmentInfo


class AlignmentCore:
    """Wrapper for core of camera alignment algorithm"""

    def generate_alignment_matrix(
        self,
        optical_control_image: numpy.typing.ArrayLike,
        reference_channel: int,
        shift_channel: int,
        magnification: int,
    ) -> typing.Tuple[numpy.typing.ArrayLike, AlignmentInfo]:
        raise NotImplementedError("generate_alignment_matrix")

    def align_image(
        self,
        alignment_matrix: numpy.typing.ArrayLike,
        image: numpy.typing.ArrayLike,
        channels_to_align: typing.List[int],
    ) -> numpy.typing.ArrayLike:
        raise NotImplementedError("align_image")

    def get_channel_name_to_index_map(self):
        raise NotImplementedError("align_image")

    def crop(self):
        raise NotImplementedError("align_image")
