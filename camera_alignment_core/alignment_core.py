import typing

from aicsimageio import AICSImage
import numpy.typing

from .alignment_info import AlignmentInfo

from .alignment_utils import \
    segment_argolight_rings as segment, \
    get_center_z, \
    crop_argolight_rings_img as crop, \
    estimate_alignment


class AlignmentCore:
    """Wrapper for core of camera alignment algorithm"""

    def generate_alignment_matrix(
        self,
        optical_control_image: numpy.typing.ArrayLike,
        reference_channel: int,
        shift_channel: int,
        magnification: int,
        px_size_xy: float,
    ) -> typing.Tuple[numpy.typing.ArrayLike, AlignmentInfo]:

        #if more than 4 dimensions, trip extra dims from beginning
        while optical_control_image.ndim > 4:
            optical_control_image = optical_control_image[0, ...]

        # detect center z-slice on reference channel
        ref_center_z, contrast = get_center_z.Executor(
            img_stack = optical_control_image[reference_channel, :, :, :]
        ).execute()

        # Crop with all available rings
        ref_crop, crop_dims, ref_labelled_grid, ref_props_grid, ref_center_cross_label, ref_number_of_rings = crop.Executor(
            img=optical_control_image[reference_channel, ref_center_z, :, :],
            pixel_size=px_size_xy,
            magnification=magnification,
            filter_px_size=50
        ).execute()

        mov_crop = optical_control_image[
                shift_channel,
                ref_center_z,
                crop_dims[0]:crop_dims[1], crop_dims[2]:crop_dims[3]
                ]

        # segment rings on reference image
        ref_seg_rings, ref_seg_rings_label, ref_props_df, ref_cross_label = segment.Executor(
            ref_crop, px_size_xy, magnification, debug_mode=True
        ).execute()

        # segment rings on moving image
        mov_seg_rings, mov_seg_rings_label, mov_props_df, mov_cross_label = segment.Executor(
            mov_crop, px_size_xy, magnification, debug_mode=True
        ).execute()

            # estimate alignment from segmentation
        tform, ref_coor_dict, transformation_parameters_dict, num_beads_for_estimation = estimate_alignment.Executor(
            ref_seg_rings, ref_seg_rings_label, ref_props_df, ref_cross_label,
            mov_seg_rings, mov_seg_rings_label, mov_props_df, mov_cross_label,
            'alignV2'
        ).execute()

        info = AlignmentInfo(
            rotation=transformation_parameters_dict['rotate_angle'],
            shift_x=transformation_parameters_dict['shift_x'],
            shift_y=transformation_parameters_dict['shift_y'],
            z_offset=0
        )

        return tform, info

        raise NotImplementedError("generate_alignment_matrix")



    def align_image(
        self,
        alignment_matrix: numpy.typing.ArrayLike,
        image: numpy.typing.ArrayLike,
        channels_to_align: typing.List[int],
    ) -> numpy.typing.ArrayLike:
        raise NotImplementedError("align_image")

    def get_channel_name_to_index_map(self, image: AICSImage) -> typing.Dict[str, int]:
        """
        Maps all channels in a split image to an index of where the channel is in the image
        Format is {"Raw channel_name" : Channel index}
        """
        channels = image.channel_names
        channel_info_dict = dict()
        for channel in channels:
            if channel in ["Bright_2", "TL_100x"]:
                channel_info_dict.update({"Raw brightfield": channels.index(channel)})
            elif channel in ["EGFP"]:
                channel_info_dict.update({"Raw 488nm": channels.index(channel)})
            elif channel in ["CMDRP"]:
                channel_info_dict.update({"Raw 638nm": channels.index(channel)})
            elif channel in ["H3342"]:
                channel_info_dict.update({"Raw 405nm": channels.index(channel)})
            elif channel in ["TaRFP"]:
                channel_info_dict.update({"Raw 561nm": channels.index(channel)})

        return channel_info_dict

    def crop(self):
        raise NotImplementedError("align_image")
