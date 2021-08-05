import typing

import numpy
from numpy.core.fromnumeric import mean
import pytest
from skimage.morphology import ball

from camera_alignment_core.alignment_utils import (
    SegmentRings,
    segment_rings,
)


class TestSegmentRings:
    def test_run(self):
        # Arrange
        ball_radius = 3

        synthetic_GT = numpy.zeros((11, 405, 611), numpy.uint16)
        for y in numpy.arange(10, 400, 30):
            for x in numpy.arange(10, 610, 30):
                synthetic_GT[
                    6 - ball_radius : 6 + ball_radius + 1,
                    y - ball_radius : y + ball_radius + 1,
                    x - ball_radius : x + ball_radius + 1,
                ] = ball(ball_radius)

        cross = numpy.zeros((11, 31, 31))
        cross[:, 13:20, :] = 1
        cross[:, :, 13:20] = 1
        cross[:2, :, :] = 0
        cross[-2:, :, :] = 0

        synthetic_GT[:, 174:205, 294:325] = cross
        synthetic_image = (synthetic_GT * 20000) + 5000
        synthetic_image += numpy.random.normal(
            loc=1, scale=0.2, size=synthetic_image.shape
        ).astype(numpy.uint16) * numpy.mean(synthetic_image.flatten()).astype(
            numpy.uint16
        )

        # Act
        seg_rings, _, _, _ = SegmentRings(
            img=numpy.max(synthetic_image, axis=0),
            pixel_size=1,
            magnification=1,
            thresh=None,
            bead_distance_um=10,
            cross_size_um=numpy.sum(numpy.max(cross, axis=0).flatten()),
            ring_radius_um=3,
        ).run()

        intersection = numpy.sum(
            numpy.logical_and(
                numpy.max(synthetic_image, axis=0) > 0,
                seg_rings > 0,
            )
        )
        union = numpy.sum(
            numpy.logical_or(
                numpy.max(synthetic_image, axis=0) > 0,
                seg_rings > 0,
            )
        )

        iou = intersection / union

        print(f"iou: {iou}")

        # Assert
        assert iou < 0.1
