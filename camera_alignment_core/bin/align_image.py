import argparse
import datetime
import logging
import pathlib
import sys
import tempfile
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
)

log = logging.getLogger(LOGGER_NAME)


class Args(argparse.Namespace):
    def parse(self, parser_args: typing.List[str]):
        parser = argparse.ArgumentParser(
            description="Run given file through camera alignment, outputting single file per scene."
        )

        parser.add_argument(
            "input_fms_file_id",
            type=str,
            help="FMS file_id for microscopy image that requires alignment",
        )

        parser.add_argument(
            "optical_control_fms_file_id",
            type=str,
            help="FMS file_id for optical control image",
        )

        parser.add_argument(
            "-m",
            "--magnification",
            choices=[20, 63, 100],
            type=int,
            required=True,
            help="Magnification at which both input_image and optical_control were acquired",
        )

        parser.add_argument(
            "-s",
            "--scene",
            type=str,
            required=False,
            help="Scene within input image to align. Takes same input as AICSImage::set_scene",
        )

        parser.add_argument(
            "-t",
            "--timepoint",
            type=int,
            required=False,
            help="On which timepoint or timepoints within file to perform the alignment",
        )

        parser.add_argument(
            "-r",
            "--ref-channel",
            type=int,
            choices=[405, 488, 561, 638],
            default=405,
            dest="reference_channel",
            help="Which channel of the optical control file to treat as the 'reference' for alignment.",
        )

        parser.add_argument(
            "-a",
            "--alignment-channel",
            type=int,
            choices=[405, 488, 561, 638],
            default=638,
            dest="alignment_channel",
            help="Which channel of the optical control file to align, relative to 'reference.'",
        )

        parser.add_argument(
            "-e",
            "--fms-env",
            type=str,
            choices=["prod", "stg", "dev"],
            default="prod",
            help="FMS env to run against. You should only set this option if you're testing output.",
        )

        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            default=False,
            dest="debug",
        )

        parser.parse_args(args=parser_args, namespace=self)

        self.reference_channel: str = Args._convert_wavelength_to_channel_name(
            self.reference_channel
        )
        self.alignment_channel: str = Args._convert_wavelength_to_channel_name(
            self.alignment_channel
        )

        return self

    @staticmethod
    def _convert_wavelength_to_channel_name(wavelength: typing.Any) -> str:
        return f"Raw {wavelength}nm"

    def print_args(self):
        """Print arguments this script is running with"""

        log.info("*" * 50)
        for attr in vars(self):
            if attr == "pg_pass":
                continue
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

    start_time = datetime.datetime.now()

    fms = FileManagementSystem(env=args.fms_env)
    input_image_fms_record = fms.find_one_by_id(args.input_fms_file_id)
    if not input_image_fms_record:
        raise ValueError(
            f"Could not find image in FMS with ID: {args.input_fms_file_id}"
        )

    control_image_fms_record = fms.find_one_by_id(args.optical_control_fms_file_id)
    if not control_image_fms_record:
        raise ValueError(
            f"Could not find optical control image in FMS with ID: {args.optical_control_fms_file_id}"
        )

    alignment_core = AlignmentCore()

    control_image = AICSImage(control_image_fms_record.path)
    control_image_channel_map = alignment_core.get_channel_name_to_index_map(
        control_image
    )

    assert (
        control_image.physical_pixel_sizes.X == control_image.physical_pixel_sizes.Y
    ), "Physical pixel sizes in X and Y dimensions do not match in optical control image"

    alignment_matrix, _ = alignment_core.generate_alignment_matrix(
        control_image.get_image_data(),
        reference_channel=control_image_channel_map[args.reference_channel],
        shift_channel=control_image_channel_map[args.alignment_channel],
        magnification=args.magnification,
        px_size_xy=control_image.physical_pixel_sizes.X,
    )

    image = AICSImage(input_image_fms_record.path)
    # Iterate over all scenes in the image...
    for scene in image.scenes:
        # ...operate on current scene
        image.set_scene(scene)

        # arrange some values for later use
        channel_name_to_index_map = alignment_core.get_channel_name_to_index_map(image)

        # align each timepoint in the image
        processed_timepoints: typing.List[numpy.typing.NDArray[numpy.uint16]] = list()
        for timepoint in range(0, image.dims.T):
            image_slice = image.get_image_data("CZYX", T=timepoint)
            processed = alignment_core.align_image(
                alignment_matrix,
                image_slice,
                channel_name_to_index_map,
                args.magnification,
            )
            processed_timepoints.append(processed)

        # collect all newly aligned timepoints into one file and save to FMS
        with tempfile.TemporaryDirectory() as tempdir:
            temp_save_path = (
                pathlib.Path(tempdir)
                / f"{pathlib.Path(input_image_fms_record.name).stem}_{scene}_aligned.ome.tiff"
            )

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

            # Save combined file to FMS
            log.debug("Uploading %s to FMS", temp_save_path)
            metadata = {
                "provenance": {
                    "input_files": [
                        args.input_fms_file_id,
                        args.optical_control_fms_file_id,
                    ],
                    "algorithm": f"camera_alignment_core v{__version__}",
                },
            }
            uploaded_file = fms.upload_file(
                temp_save_path, file_type="image", metadata=metadata
            )
            log.info(
                "Uploaded aligned scene (%s) as FMS file_id %s", scene, uploaded_file.id
            )

    end_time = datetime.datetime.now()
    log.info(f"Finished in {end_time - start_time}")


if __name__ == "__main__":
    main()
