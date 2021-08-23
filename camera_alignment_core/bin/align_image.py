import argparse
import logging
import pathlib
import shutil
import sys
import tempfile
import time
import typing

from aicsfiles import FileManagementSystem
from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
import numpy
import numpy.typing

from camera_alignment_core import (
    AlignmentCore,
    __version__,
)
from camera_alignment_core.constants import (
    LOGGER_NAME,
    Channel,
    Magnification,
)

from .image_dimension_action import (
    ImageDimensionAction,
)

log = logging.getLogger(LOGGER_NAME)


class Args(argparse.Namespace):
    def parse(self, parser_args: typing.List[str]):
        parser = argparse.ArgumentParser(
            description="Run given file through camera alignment, outputting single file per scene."
        )

        parser.add_argument(
            "image",
            type=str,
            help="Microscopy image that requires alignment. Can specify as either an FMS id or as a file path.",
        )

        parser.add_argument(
            "optical_control",
            type=str,
            help="Optical control image to use to align `image`. Can specify as either an FMS id or as a file path.",
        )

        parser.add_argument(
            "--magnification",
            choices=[mag.value for mag in list(Magnification)],
            type=int,
            required=True,
            help="Magnification at which both `image` and `optical_control` were acquired.",
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
            "--fms-env",
            type=str,
            choices=["prod", "stg", "dev"],
            default="prod",
            help="FMS env to run against.",
        )

        parser.add_argument(
            "--out-dir",
            type=lambda p: pathlib.Path(p).expanduser().resolve(strict=True),
            help="If provided, aligned images will be saved into `out-dir` instead of uploaded to FMS.",
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

    fms = FileManagementSystem(env=args.fms_env)

    # Check if args.image is a file path, else treat as an FMS ID
    if pathlib.Path(args.image).exists():
        input_image_path = pathlib.Path(args.image)
    else:
        input_image_fms_record = fms.find_one_by_id(args.image)
        if not input_image_fms_record:
            raise ValueError(f"Could not find image in FMS with ID: {args.image}")
        input_image_path = pathlib.Path(input_image_fms_record.path)

    # Check if args.optical_control is a file path, else treat as an FMS ID
    if pathlib.Path(args.optical_control).exists():
        control_image_path = pathlib.Path(args.optical_control)
    else:
        control_image_fms_record = fms.find_one_by_id(args.optical_control)
        if not control_image_fms_record:
            raise ValueError(
                f"Could not find optical control image in FMS with ID: {args.optical_control}"
            )
        control_image_path = pathlib.Path(control_image_fms_record.path)

    alignment_core = AlignmentCore()

    control_image = AICSImage(control_image_path)
    control_image_channel_info = alignment_core.get_channel_info(control_image)

    assert (
        control_image.physical_pixel_sizes.X == control_image.physical_pixel_sizes.Y
    ), "Physical pixel sizes in X and Y dimensions do not match in optical control image"

    alignment_matrix, _ = alignment_core.generate_alignment_matrix(
        control_image.get_image_data("CZYX", T=0),
        reference_channel=control_image_channel_info.index_of_channel(
            args.reference_channel
        ),
        shift_channel=control_image_channel_info.index_of_channel(
            args.alignment_channel
        ),
        magnification=args.magnification,
        px_size_xy=control_image.physical_pixel_sizes.X,
    )

    image = AICSImage(input_image_path)
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
        with tempfile.TemporaryDirectory() as tempdir:
            # In general, expect multi-scene images as input. Input may, however, be single scene image.
            # In the case of a single scene image file, **assume** the filename already contains the scene name, e.g. "3500004473_100X_20210430_1c-Scene-24-P96-G06.czi."
            # Unfortunately, cannot check `if scene in input_image_path.stem`--that assumes too much conformance between how the scene is named
            # in the filename and how AICSImageIO deals with scene naming.
            stem, *_ = input_image_path.name.split(".")
            out_name = (
                f"{stem}_aligned.ome.tiff"
                if len(image.scenes) == 1
                else f"{stem}_Scene-{scene}_aligned.ome.tiff"
            )
            temp_save_path = pathlib.Path(tempdir) / out_name

            processed_image_data = numpy.stack(processed_timepoints)  # TCZYX
            (T, C, Z, Y, X) = processed_image_data.shape
            # TODO: Uncomment once https://github.com/AllenCellModeling/aicsimageio/pull/292 merges
            # ome_xml = image.ome_metadata
            # ome_xml.images[0].pixels.size_t = T
            # ome_xml.images[0].pixels.size_c = C
            # ome_xml.images[0].pixels.size_z = Z
            # ome_xml.images[0].pixels.size_y = Y
            # ome_xml.images[0].pixels.size_x = X
            # ome_xml.images[0].pixels.dimension_order = DimensionOrder.XYZCT
            OmeTiffWriter.save(
                data=processed_image_data,
                # TODO, same as above, uncomment once https://github.com/AllenCellModeling/aicsimageio/pull/292 merges
                # ome_xml=ome_xml,
                uri=temp_save_path,
                channel_names=image.channel_names,
                physical_pixel_sizes=PhysicalPixelSizes(Z=Z, Y=Y, X=X),
                dim_order="TCZYX",
            )

            # If args.out_dir is specified, save output to out_dir
            if args.out_dir:
                shutil.copy(temp_save_path, args.out_dir)
                log.debug("Copied %s to %s", temp_save_path, args.out_dir)
            else:
                # Save combined file to FMS
                log.debug("Uploading %s to FMS", temp_save_path)
                metadata = {
                    "provenance": {
                        "input_files": [
                            input_image_fms_record.id,
                            control_image_fms_record.id,
                        ],
                        "algorithm": f"camera_alignment_core v{__version__}",
                    },
                }
                uploaded_file = fms.upload_file(
                    temp_save_path, file_type="image", metadata=metadata
                )
                log.info(
                    "Uploaded aligned scene %s for file %s as FMS file_id %s",
                    scene,
                    args.image,
                    uploaded_file.id,
                )

    end_time = time.perf_counter()
    log.info(f"Finished in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    main()
