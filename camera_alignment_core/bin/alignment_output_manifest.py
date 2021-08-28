import dataclasses
import json
import pathlib
import typing


class PathlibAwareEncoder(json.JSONEncoder):
    def default(self, obj: typing.Any) -> typing.Any:
        if isinstance(obj, pathlib.Path):
            return str(obj)

        return super().default(obj)


@dataclasses.dataclass
class AlignedImage:
    # Which scene from the original, unaligned image this corresponds to
    scene: int

    # Output path of the aligned image
    path: pathlib.Path


@dataclasses.dataclass
class AlignmentOutputManifest:
    """
    This provides a schema for the descriptive output file of the `align` console script.
    It also provides a utilities for serializing to and deserializing from that file.
    """

    DEFAULT_FILE_NAME_PATTERN = "{year}_{month}_{day}-alignment_output.json"

    @staticmethod
    def from_file(path: pathlib.Path) -> "AlignmentOutputManifest":
        deserialized = json.loads(path.read_text())
        aligned_images = [
            AlignedImage(scene=image["scene"], path=pathlib.Path(image["path"]))
            for image in deserialized.get("aligned_images", [])
        ]
        return AlignmentOutputManifest(
            alignment_info_path=pathlib.Path(deserialized["alignment_info_path"]),
            aligned_optical_control_path=pathlib.Path(
                deserialized["aligned_optical_control_path"]
            ),
            aligned_images=aligned_images,
        )

    # Serialized AlignmentInfo. Records translation, rotation, and scaling done as part of alignment.
    alignment_info_path: pathlib.Path

    # The aligned optical control. This is performed and output as a reference.
    # I.e., can be used to easily inspect how well the alignment worked.
    aligned_optical_control_path: pathlib.Path

    # One or many aligned images. The input microscopy image will be split by scene.
    aligned_images: typing.List[AlignedImage]

    def to_file(self, path: pathlib.Path) -> None:
        as_dict = dataclasses.asdict(self)
        path.write_text(json.dumps(as_dict, cls=PathlibAwareEncoder, indent=4))
