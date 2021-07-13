import typing

import os

import numpy.typing

from aicsimageio import AICSImage

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

    def get_channel_name_to_index_map(
        self,
        im_path: os.typing.pathLike
    ) -> typing.Dict[str, int]:
        # Maps all channels in a split image to an index of where the channel is in the image
        # Format is {"Raw channel_name" : Channel index}

        im = AICSImage(im_path)
        channels = im.channel_names
        channel_info_dict = dict()
        for channel in channels:
            if channel in ['Bright_2', 'TL_100x']:
                channel_info_dict.update({'Raw brightfield': channels.index(channel)})
            elif channel in ['EGFP']:
                channel_info_dict.update({'Raw 488nm': channels.index(channel)})
            elif channel in ['CMDRP']:
                channel_info_dict.update({'Raw 638nm': channels.index(channel)})
            elif channel in ['H3342']:
                channel_info_dict.update({'Raw 405nm': channels.index(channel)})
            elif channel in ['TaRFP']:
                channel_info_dict.update({'Raw 561nm': channels.index(channel)})

        return channel_info_dict

    def crop(self):
        raise NotImplementedError("align_image")
