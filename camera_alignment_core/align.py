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
    get_channel_info,
)
from .alignment_utils import AlignmentInfo
from .constants import (
    LOGGER_NAME,
    Channel,
    Magnification,
)

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
    >>>     reference_channel=Channel.RAW_561_NM,
    >>>     alignment_channel=Channel.RAW_638_NM,
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
        reference_channel: Channel,
        alignment_channel: Channel,
        out_dir: pathlib.Path,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        optical_control : Union[str, Path]
            Optical control image that will be used to generate an alignment matrix. Passed as-is to aicsimageio.AICSImage constructor.
        magnification : Magnification
            Magnification at which `optical_control` (and any images to be aligned using `optical_control`) was acquired.
        reference_channel : Channel
            Which channel of `optical_control` to treat as the 'reference' for alignment. I.e., the 'static' channel.
            Defined in terms of the wavelength used in that channel.
        alignment_channel : Channel
            Which channel of `optical_control` to align, relative to 'reference.' I.e., the 'moving' channel.
            Defined in terms of the wavelength used in that channel.
        out_dir : pathlib.Path
            Path to directory in which to save file output of alignment. Neither the directory nor its parents
            need to exist prior to running (though it's OK if they do); directories will be created if not.
        """
        self._optical_control_path = pathlib.Path(optical_control)
        self._optical_control = AICSImage(optical_control)

        self._magnification = magnification
        self._reference_channel = reference_channel
        self._alignment_channel = alignment_channel
        self._out_dir = out_dir

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
            control_image_channel_info = get_channel_info(self._optical_control)

            assert (
                self._optical_control.physical_pixel_sizes.X
                == self._optical_control.physical_pixel_sizes.Y
            ), "Physical pixel sizes in X and Y dimensions do not match in optical control image"

            control_image_data = self._optical_control.get_image_data("CZYX", T=0)
            alignment_matrix, alignment_info = generate_alignment_matrix(
                control_image_data,
                reference_channel=control_image_channel_info.index_of_channel(
                    self._reference_channel
                ),
                shift_channel=control_image_channel_info.index_of_channel(
                    self._alignment_channel
                ),
                magnification=self._magnification.value,
                px_size_xy=self._optical_control.physical_pixel_sizes.X,
            )

            self._alignment_matrix = alignment_matrix
            self._alignment_info = alignment_info

        return AlignmentTransform(self._alignment_matrix, self._alignment_info)

    def align_optical_control(self, crop_output: bool = True) -> pathlib.Path:
        """Align the optical control image using the similarity matrix generated from
        the optical control itself. Useful as a reference for judging the quality of the alignment.

        Keyword Arguments
        ----------
        crop_output : Optional[bool]
            Optionally do not crop aligned image according to standard dimensions
            for the magnification at which the image was acquired. Defaults to cropping.

        Returns
        -------
        pathlib.Path

        Notes
        -----
        This method will output the aligned optical control image to a file as a side-effect,
        returning the pathlib.Path to the file.
        """
        aligned_control = align_image(
            self.alignment_transform.matrix,
            self._optical_control.get_image_data("CZYX", T=0),
            get_channel_info(self._optical_control),
            self._magnification.value,
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

        Keyword Arguments
        -----------------
        scenes : Optional[List[int]]
            On which scene or scenes within `image` to align. If not specified, will align all scenes within `image`.
            Specify as list of 0-index scene indices within `image`.
        timepoints : Optional[List[int]]
            On which timepoint or timepoints within `image` to perform the alignment. If not specified, will align all timepoints within `image`.
            Specify as list of 0-index timepoint indices within `image`.
        crop_output : Optional[bool]
            Optionally do not crop aligned image according to standard dimensions
            for the magnification at which the image was acquired. Defaults to cropping.

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
            start_time_scene = time.perf_counter()

            # Operate on current scene
            aics_image.set_scene(scene)

            channel_info = get_channel_info(aics_image)

            # Align timepoints within scene
            processed_timepoints: typing.List[
                numpy.typing.NDArray[numpy.uint16]
            ] = list()
            timepoint_indices = (
                timepoints if timepoints else range(0, aics_image.dims.T)
            )
            for timepoint in timepoint_indices:
                start_time_timepoint = time.perf_counter()

                image_slice = aics_image.get_image_data("CZYX", T=timepoint)
                processed = align_image(
                    self.alignment_transform.matrix,
                    image_slice,
                    channel_info,
                    self._magnification.value,
                )
                if crop_output:
                    processed_timepoints.append(crop(processed, self._magnification))
                else:
                    processed_timepoints.append(processed)

                end_time_timepoint = time.perf_counter()
                log.debug(
                    f"END TIMEPOINT: aligned timepoint {timepoint} in {end_time_timepoint - start_time_timepoint:0.4f} seconds"
                )

            end_time_scene = time.perf_counter()
            log.debug(
                f"END SCENE: aligned scene {scene} in {end_time_scene - start_time_scene:0.4f} seconds"
            )

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
