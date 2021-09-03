import argparse
import typing


def is_empty_or_none(val: typing.Any) -> bool:
    if val is None:
        return True

    if isinstance(val, str):
        return val.strip() == ""

    return False


def parse_dimension_str(dim_str: str) -> typing.List[int]:
    """
    Take str specification for image dimension (scene or timepoint) and parse it into
    a concrete range of ints that can be used to index into image data.
    Several examples of expected input and expected output:
        - "2" -> [2]
        - "1-30" -> [1, 2, ..., 29, 30]
        - "0-21" -> [0, 1, ..., 20, 21]
        - "8-10, 13, 15-17" -> [8, 9, 10, 13, 15, 16, 17]
    """

    # E.g. ['8- 10', '9', '15 - 17']
    seperated_by_comma = [item.strip() for item in dim_str.split(",")]

    # Guard against empty ranges, e.g., "1, ,2-3,"
    compacted = filter(lambda item: item, seperated_by_comma)

    # E.g. [['8', ' 10'], ['9'], ['15 ', ' 17']]
    sub_seperated_by_dash = [item.split("-") for item in compacted]

    indices: typing.Set[int] = set()
    for input_range in sub_seperated_by_dash:
        massaged = [int(item.strip()) for item in input_range]
        if len(massaged) == 2:
            if any([is_empty_or_none(bound) for bound in massaged]):
                # Likely got input like "-17" or "17-"
                raise ValueError(f"Unable to parse range {'-'.join(input_range)}")
            start, end = massaged
            r = range(start, end + 1)
            indices.update(r)
        elif len(massaged) == 1:
            indices.add(massaged[0])
        else:
            raise ValueError(f"Unable to parse range {input_range}")

    return sorted(indices)


class ImageDimensionAction(argparse.Action):
    """
    Custom argparse action that can parse image dimensions (e.g., timepoints and scenes)
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: typing.Any,
        option_string: typing.Optional[str] = None,
    ):
        if values and isinstance(values, str):
            setattr(namespace, self.dest, parse_dimension_str(values))
