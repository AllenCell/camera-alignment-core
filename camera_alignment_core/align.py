import logging
import pathlib
import time
import typing

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import numpy
import numpy.typing

from .alignment_core import (
    align_image,
    crop,
    generate_alignment_matrix,
)
from .alignment_utils import AlignmentInfo
from .channel_info import ChannelInfo
from .constants import LOGGER_NAME, Magnification

log = logging.getLogger(LOGGER_NAME)


class AlignmentTransform(typing.NamedTuple):
    matrix: numpy.typing.NDArray[numpy.float16]
    info: AlignmentInfo


class AlignedImage(typing.NamedTuple):
    # Which scene from the original, unaligned image this corresponds to
    scene: int

    # Output path of the aligned image
    path: pathlib.Path


class Align:
    """High-level API for core camera alignment functionality.

    Example
    -------
    >>> align = Align(
    >>>     optical_control="/some/path/to/an/argolight-field-of-rings.czi",
    >>>     magnification=Magnification(20),
    >>>     out_dir="/tmp/whereever",
    >>> )
    >>> aligned_scenes = align.align_image("/some/path/to/an/image.czi")
    >>> aligned_optical_control = align.align_optical_control()
    >>> alignment_matrix = align.alignment_transform.matrix
    >>> alignment_info = align.alignment_transform.info
    """

    def __init__(
        self,
        optical_control: typing.Union[str, pathlib.Path],
        magnification: Magnification,
        out_dir: typing.Union[str, pathlib.Path],
        reference_channel_index: typing.Optional[int] = None,
        alignment_channel_index: typing.Optional[int] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        optical_control : Union[str, Path]
            Optical control image that will be used to generate an alignment matrix. Passed as-is to aicsimageio.AICSImage constructor.
        magnification : Magnification
            Magnification at which `optical_control` (and any images to be aligned using `optical_control`) was acquired.
        out_dir : Union[str, Path]
            Path to directory in which to save file output of alignment. Neither the directory nor its parents
            need to exist prior to running (though it's OK if they do); directories will be created if not.

        Keyword Arguments
        -----------------
        reference_channel_index : int
            Which channel of `optical_control` to treat as the 'reference' for alignment. I.e., the 'static' channel.
            Defined in terms of the wavelength used in that channel.
        alignment_channel_index : int
            Which channel of `optical_control` to align, relative to 'reference.' I.e., the 'moving' channel.
            Defined in terms of the wavelength used in that channel.
        """
        self._optical_control_path = pathlib.Path(optical_control)
        self._optical_control = AICSImage(optical_control)

        self._magnification = magnification
        self._out_dir = pathlib.Path(out_dir)

        self._reference_channel_index = reference_channel_index
        self._alignment_channel_index = alignment_channel_index

        self._alignment_matrix: typing.Optional[
            numpy.typing.NDArray[numpy.float16]
        ] = None
        self._alignment_info: typing.Optional[AlignmentInfo] = None

    @property
    def alignment_transform(self) -> AlignmentTransform:
        """
        Get the similarity matrix and camera_alignment_core.utils.AlignmentInfo used to perform camera alignment.
        """
        if self._alignment_matrix is None or self._alignment_info is None:
            assert (
                self._optical_control.physical_pixel_sizes.X
                == self._optical_control.physical_pixel_sizes.Y
            ), "Physical pixel sizes in X and Y dimensions do not match in optical control image"

            # If the reference channel and/or alignment channel were not specified,
            # query the image metadata to find the channels closest in their emission wavelength
            # between the two cameras. According to Nathalie (2021-11), it doesn't matter
            # which is set as the reference and which is set as the alignment.
            if not self._reference_channel_index or not self._alignment_channel_index:
                channel_info = ChannelInfo(
                    self._optical_control, self._optical_control_path
                )
                channels_close_in_emission_wavelength = (
                    channel_info.find_channels_closest_in_emission_wavelength_across_cameras()
                )
                self._reference_channel_index = channels_close_in_emission_wavelength[
                    0
                ].channel_index
                self._alignment_channel_index = channels_close_in_emission_wavelength[
                    1
                ].channel_index

            control_image_data = self._optical_control.get_image_data("CZYX", T=0)
            alignment_matrix, alignment_info = generate_alignment_matrix(
                control_image_data,
                reference_channel=self._reference_channel_index,
                shift_channel=self._alignment_channel_index,
                magnification=self._magnification.value,
                px_size_xy=self._optical_control.physical_pixel_sizes.X,
            )

            self._alignment_matrix = alignment_matrix
            self._alignment_info = alignment_info

        return AlignmentTransform(self._alignment_matrix, self._alignment_info)

    def align_optical_control(
        self, channels_to_align: typing.List[int], crop_output: bool = True
    ) -> pathlib.Path:
        """Align the optical control image using the similarity matrix generated from
        the optical control itself. Useful as a reference for judging the quality of the alignment.

        Parameters
        ----------
        channels_to_align : List[int]
            Index positions of channels within `image` that should be aligned. N.b.: indices start at 0.
            E.g.: Specify [0, 3] to align the channels at index positions 0 and 3 within `image`.

        Keyword Arguments
        ----------
        crop_output : Optional[bool]
            Optional flag for toggling whether to crop aligned image according to standard dimensions
            for the magnification at which the image was acquired. Defaults to `True`, which means,
            "yes, crop the image."

        Returns
        -------
        pathlib.Path

        Notes
        -----
        This method will output the aligned optical control image to a file as a side-effect,
        returning the pathlib.Path to the file.
        """
        aligned_control = align_image(
            self._optical_control.get_image_data("CZYX", T=0),
            self.alignment_transform.matrix,
            channels_to_align,
        )

        if crop_output:
            aligned_control = crop(aligned_control, self._magnification)

        aligned_control_outpath = (
            self._out_dir / f"{self._optical_control_path.stem}_aligned.ome.tiff"
        )
        OmeTiffWriter.save(
            # aligned_control is CZYX, wrap in an array to fill it out to TCZYX
            data=numpy.stack([aligned_control]),
            uri=aligned_control_outpath,
            channel_names=self._optical_control.channel_names,
            dim_order="TCZYX",
        )
        return aligned_control_outpath

    def align_image(
        self,
        image: typing.Union[str, pathlib.Path],
        channels_to_align: typing.List[int],
        scenes: typing.List[int] = [],
        timepoints: typing.List[int] = [],
        crop_output: bool = True,
    ) -> typing.List[AlignedImage]:
        """Align `image` using similarity transform generated from the optical control image passed to
        this instance at construction.

        Parameters
        ----------
        image : Union[str, Path]
            Microscopy image that requires alignment. Passed as-is to aicsimageio.AICSImage constructor.
        channels_to_align : List[int]
            Index positions of channels within `image` that should be aligned. N.b.: indices start at 0.
            E.g.: Specify [0, 3] to align the channels at index positions 0 and 3 within `image`.

        Keyword Arguments
        -----------------
        scenes : Optional[List[int]]
            On which scene or scenes within `image` to align. If not specified, will align all scenes within `image`.
            Specify as list of 0-index scene indices within `image`.
        timepoints : Optional[List[int]]
            On which timepoint or timepoints within `image` to perform the alignment. If not specified, will align all timepoints within `image`.
            Specify as list of 0-index timepoint indices within `image`.
        crop_output : Optional[bool]
            Optional flag for toggling whether to crop aligned image according to standard dimensions
            for the magnification at which the image was acquired. Defaults to `True`, which means,
            "yes, crop the image."

        Returns
        -------
        List[AlignedImage]
            A list of namedtuples, each of which describes a scene within `image` that was aligned.
        """
        aics_image = AICSImage(image)

        aligned_scenes: typing.List[AlignedImage] = []

        # Iterate over scenes to align
        scene_indices = scenes if scenes else range(len(aics_image.scenes))
        for scene in scene_indices:
            # Operate on current scene
            aics_image.set_scene(scene)

            # Align timepoints within scene
            processed_timepoints: typing.List[
                numpy.typing.NDArray[numpy.uint16]
            ] = list()
            timepoint_indices = (
                timepoints if timepoints else range(0, aics_image.dims.T)
            )
            for timepoint in timepoint_indices:
                image_slice = aics_image.get_image_data("CZYX", T=timepoint)
                processed = align_image(
                    image_slice, self.alignment_transform.matrix, channels_to_align
                )
                if crop_output:
                    processed_timepoints.append(crop(processed, self._magnification))
                else:
                    processed_timepoints.append(processed)

                log.debug(f"END TIMEPOINT: aligned timepoint {timepoint}")

            log.debug(f"END SCENE: aligned scene {scene}")

            # Collect all newly aligned timepoints for this scene into one file and save output
            # In general, expect multi-scene images as input. Input may, however, be single scene image.
            # In the case of a single scene image file, **assume** the filename already contains the scene name, e.g. "3500004473_100X_20210430_1c-Scene-24-P96-G06.czi."
            # Unfortunately, cannot check `if scene in input_image_path.stem`--that assumes too much conformance between how the scene is named
            # in the filename and how AICSImageIO deals with scene naming.
            stem, *_ = pathlib.Path(image).name.split(".")
            out_name = (
                f"{stem}_aligned.ome.tiff"
                if len(aics_image.scenes) == 1
                else f"{stem}_Scene-{scene}_aligned.ome.tiff"
            )
            save_path = pathlib.Path(self._out_dir) / out_name
            processed_image_data = numpy.stack(processed_timepoints)  # TCZYX
            OmeTiffWriter.save(
                data=processed_image_data,
                uri=save_path,
                channel_names=aics_image.channel_names,
                dim_order="TCZYX",
            )
            aligned_scenes.append(AlignedImage(scene, save_path))

        return aligned_scenes
