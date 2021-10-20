import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time
import typing

from camera_alignment_core import Align
from camera_alignment_core.alignment_output_manifest import (
    AlignmentOutputManifest,
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
    align = Align(
        optical_control=args.optical_control,
        magnification=Magnification(args.magnification),
        reference_channel=args.reference_channel,
        alignment_channel=args.alignment_channel,
        out_dir=args.out_dir,
    )

    # Align the optical itself as a control
    aligned_control_outpath = align.align_optical_control(crop_output=args.crop)

    # Align the image
    aligned_images = align.align_image(
        args.image,
        scenes=args.scene,
        timepoints=args.timepoint,
        crop_output=args.crop,
    )

    # Output alignment info as JSON
    control_image_name = pathlib.Path(args.optical_control).stem
    alignment_info_outpath = args.out_dir / f"{control_image_name}_info.json"
    alignment_info_outpath.write_text(
        json.dumps(dataclasses.asdict(align.alignment_transform.info), indent=4)
    )

    # Save file describing/recording output of this script
    output = AlignmentOutputManifest(
        alignment_info_outpath, aligned_control_outpath, aligned_images
    )
    output.to_file(out_path=args.manifest_file, out_dir=args.out_dir)

    end_time = time.perf_counter()
    log.info(f"Finished in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    main()
