import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import typing

from aicsfiles import FileManagementSystem
from aicsimageio import AICSImage
from alignment_core import AlignmentCore
import numpy.typing

log = logging.getLogger(__name__)


class Args(argparse.Namespace):
    def parse(self, args=None):
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
            dest="ref_channel",
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

        parser_args = args if args else sys.argv[1:]
        parser.parse_args(args=parser_args, namespace=self)

        return self

    def print_args(self):
        """Print arguments this script is running with"""

        log.info("*" * 50)
        for attr in vars(self):
            if attr == "pg_pass":
                continue
            log.info(f"{attr}: {getattr(self, attr)}")
        log.info("*" * 50)

    def _ensure_parent_exists(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def main():
    args = Args().parse()

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)4s:%(name)s %(filename)s %(lineno)4s %(asctime)s] %(message)s",
        handlers=[logging.StreamHandler()],  # log to stderr
    )
    if args.debug:
        logging.root.setLevel(logging.DEBUG)

    args.print_args()

    start_time = datetime.now()

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
    alignment_matrix, _ = alignment_core.generate_alignment_matrix(
        control_image.get_image_data(),
        reference_channel=control_image_channel_map[args.ref_channel],
        channel_to_align=control_image_channel_map[args.alignment_channel],
        magnification=args.magnification,
    )

    image = AICSImage(input_image_fms_record.path)
    # Iterate over all scenes in the image...
    for scene in image.scenes:
        # ...operate on current scene
        image.set_scene(scene)

        # arrange some values for later use
        channel_name_to_index_map = alignment_core.get_channel_name_to_index_map(image)

        # ... align each timepoint in the image
        processed_timepoints: typing.List[numpy.typing.NDArray[numpy.uint16]] = list()
        for timepoint in range(0, image.dims.T):
            sub_selection = image.get_image_data(T=timepoint)
            processed = alignment_core.align_image(
                alignment_matrix, sub_selection, channel_name_to_index_map
            )
            processed_timepoints.append(processed)

        # stack all timepoints into new image

        # save resulting file to FMS

    end_time = datetime.now()
    log.info(f"Finished in {end_time - start_time}")


if __name__ == "__main__":
    main()
