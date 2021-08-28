import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
import sys
import time
import typing

from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
import numpy
import numpy.typing

from camera_alignment_core import AlignmentCore
from camera_alignment_core.constants import (
    LOGGER_NAME,
    Channel,
    Magnification,
)

from .alignment_output_manifest import (
    AlignedImage,
    AlignmentOutputManifest,
)
from .image_dimension_action import (
    ImageDimensionAction,
)

log = logging.getLogger(LOGGER_NAME)


class Args(argparse.Namespace):
    @property
    def crop(self) -> bool:
        """
        The purpose of this property is to avoid the use of double-negatives elsewhere.
        The console_script accepts a "--no-crop" optional argument which defaults to False. So:
            - if self.no_crop is truthy, self.crop should be False
            - if self.no_crop is falsey, self.crop should be True
        """
        if getattr(self, "no_crop"):
            return False

        return True

    def parse(self, parser_args: typing.List[str]):
        parser = argparse.ArgumentParser(
            description="Run given file through camera alignment, outputting single file per scene."
        )

        parser.add_argument(
            "image",
            type=str,
            help="Microscopy image that requires alignment. Passed directly to aicsimageio.AICSImage constructor.",
        )

        parser.add_argument(
            "optical_control",
            type=str,
            help="Optical control image to use to align `image`. Passed directly to aicsimageio.AICSImage constructor.",
        )

        parser.add_argument(
            "--out-dir",
            dest="out_dir",
            required=True,
            type=lambda p: pathlib.Path(p).expanduser().resolve(strict=True),
            help="Save output into `out-dir`",
        )

        parser.add_argument(
            "--magnification",
            choices=[mag.value for mag in list(Magnification)],
            type=int,
            required=True,
            help="Magnification at which both `image` and `optical_control` were acquired.",
        )

        parser.add_argument(
            "--manifest-file",
            dest="manifest_file",
            required=False,
            type=lambda p: pathlib.Path(p).expanduser().resolve(),
            help="Path to file at which manifest of output of this script will be written. See camera_alignment_core.bin.alignment_output_manifest.",
        )

        parser.add_argument(
            "--scene",
            type=str,
            required=False,
            dest="scene",
            action=ImageDimensionAction,
            help="On which scene or scenes within `image` to align. If not specified, will align all scenes within `image`.",
        )

        parser.add_argument(
            "--timepoint",
            type=str,
            required=False,
            dest="timepoint",
            action=ImageDimensionAction,
            help="On which timepoint or timepoints within `image` to perform the alignment. If not specified, will align all timepoints within `image`.",
        )

        parser.add_argument(
            "--ref-channel",
            type=int,
            choices=[405, 488, 561, 638],
            default=561,
            dest="reference_channel",
            help=(
                "Which channel of `optical_control` to treat as the 'reference' for alignment. I.e., the 'static' channel. Defined in terms of the wavelength used in that channel."
            ),
        )

        parser.add_argument(
            "--align-channel",
            type=int,
            choices=[405, 488, 561, 638],
            default=638,
            dest="alignment_channel",
            help=(
                "Which channel of `optical_control` to align, relative to 'reference.' I.e., the 'moving' channel. Defined in terms of the wavelength used in that channel."
            ),
        )

        parser.add_argument(
            "--no-crop",
            action="store_true",  # creates default value of False
            dest="no_crop",
            help="Do not to crop the aligned image(s).",
        )

        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            default=False,
            dest="debug",
        )

        parser.parse_args(args=parser_args, namespace=self)

        self.reference_channel: Channel = Args._convert_wavelength_to_channel_name(
            self.reference_channel
        )
        self.alignment_channel: Channel = Args._convert_wavelength_to_channel_name(
            self.alignment_channel
        )

        return self

    @staticmethod
    def _convert_wavelength_to_channel_name(wavelength: typing.Any) -> Channel:
        return Channel(f"Raw {wavelength}nm")

    def print_args(self):
        """Print arguments this script is running with"""
        log.info("*" * 50)
        for attr in vars(self):
            log.info(f"{attr}: {getattr(self, attr)}")
        log.info("*" * 50)


def save_ndarray_to_ome_tiff(
    data: numpy.typing.NDArray[numpy.uint16],
    save_path: pathlib.Path,
    channel_names: typing.List[str],
) -> None:
    (T, C, Z, Y, X) = data.shape
    # TODO: Uncomment once https://github.com/AllenCellModeling/aicsimageio/pull/292 merges
    # ome_xml = image.ome_metadata
    # ome_xml.images[0].pixels.size_t = T
    # ome_xml.images[0].pixels.size_c = C
    # ome_xml.images[0].pixels.size_z = Z
    # ome_xml.images[0].pixels.size_y = Y
    # ome_xml.images[0].pixels.size_x = X
    # ome_xml.images[0].pixels.dimension_order = DimensionOrder.XYZCT
    OmeTiffWriter.save(
        data=data,
        # TODO, same as above, uncomment once https://github.com/AllenCellModeling/aicsimageio/pull/292 merges
        # ome_xml=ome_xml,
        uri=save_path,
        channel_names=channel_names,
        physical_pixel_sizes=PhysicalPixelSizes(Z=Z, Y=Y, X=X),
        dim_order="TCZYX",
    )
    log.debug("Output %s", save_path)


def main(cli_args: typing.List[str] = sys.argv[1:]):
    args = Args().parse(cli_args)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)4s:%(name)s %(filename)s %(lineno)4s %(asctime)s] %(message)s",
        handlers=[logging.StreamHandler()],  # log to stderr
    )
    if args.debug:
        logging.root.setLevel(logging.DEBUG)

    args.print_args()

    start_time = time.perf_counter()

    alignment_core = AlignmentCore()

    control_image = AICSImage(args.optical_control)
    control_image_channel_info = alignment_core.get_channel_info(control_image)

    assert (
        control_image.physical_pixel_sizes.X == control_image.physical_pixel_sizes.Y
    ), "Physical pixel sizes in X and Y dimensions do not match in optical control image"

    control_image_data = control_image.get_image_data("CZYX", T=0)
    alignment_matrix, alignment_info = alignment_core.generate_alignment_matrix(
        control_image_data,
        reference_channel=control_image_channel_info.index_of_channel(
            args.reference_channel
        ),
        shift_channel=control_image_channel_info.index_of_channel(
            args.alignment_channel
        ),
        magnification=args.magnification,
        px_size_xy=control_image.physical_pixel_sizes.X,
    )

    # Output alignment info as JSON
    control_image_name = pathlib.Path(args.optical_control).stem
    alignment_info_outpath = (
        pathlib.Path(args.out_dir) / f"{control_image_name}_info.json"
    )
    alignment_info_outpath.write_text(
        json.dumps(dataclasses.asdict(alignment_info), indent=4)
    )

    # Align the optical itself as a control
    aligned_control = alignment_core.align_image(
        alignment_matrix,
        control_image_data,
        alignment_core.get_channel_info(control_image),
        args.magnification,
        crop=args.crop,
    )
    aligned_control_outpath = (
        pathlib.Path(args.out_dir) / f"{control_image_name}_aligned.ome.tiff"
    )
    save_ndarray_to_ome_tiff(
        # aligned_control is CZYX, wrap in an array to fill it out to TCZYX
        numpy.stack([aligned_control]),
        aligned_control_outpath,
        control_image.channel_names,
    )

    image = AICSImage(args.image)
    aligned_image_paths: typing.List[AlignedImage] = []

    # Iterate over scenes to align
    scene_indices = args.scene if args.scene else range(len(image.scenes))
    for scene in scene_indices:
        start_time_scene = time.perf_counter()

        # Operate on current scene
        image.set_scene(scene)

        # Align timepoints within scene
        processed_timepoints: typing.List[numpy.typing.NDArray[numpy.uint16]] = list()
        timepoint_indices = args.timepoint if args.timepoint else range(0, image.dims.T)
        for timepoint in timepoint_indices:
            start_time_timepoint = time.perf_counter()

            image_slice = image.get_image_data("CZYX", T=timepoint)
            processed = alignment_core.align_image(
                alignment_matrix,
                image_slice,
                alignment_core.get_channel_info(image),
                args.magnification,
                crop=args.crop,
            )
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
        stem, *_ = pathlib.Path(args.image).name.split(".")
        out_name = (
            f"{stem}_aligned.ome.tiff"
            if len(image.scenes) == 1
            else f"{stem}_Scene-{scene}_aligned.ome.tiff"
        )
        save_path = pathlib.Path(args.out_dir) / out_name
        processed_image_data = numpy.stack(processed_timepoints)  # TCZYX
        save_ndarray_to_ome_tiff(processed_image_data, save_path, image.channel_names)
        aligned_image_paths.append(AlignedImage(scene, save_path))

    # Save file describing/recording output of this script
    output = AlignmentOutputManifest(
        alignment_info_outpath, aligned_control_outpath, aligned_image_paths
    )
    today = datetime.date.today()
    default_manifest_file_path = (
        args.out_dir
        / AlignmentOutputManifest.DEFAULT_FILE_NAME_PATTERN.format(
            year=today.year, month=today.month, day=today.day
        )
    )
    manifest_file_path = (
        args.manifest_file if args.manifest_file else default_manifest_file_path
    )
    output.to_file(manifest_file_path)

    end_time = time.perf_counter()
    log.info(f"Finished in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    main()
