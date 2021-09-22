import math

import numpy
from numpy.core.fromnumeric import mean
import pandas
from skimage.measure import label, regionprops
from skimage.morphology import ball

from camera_alignment_core.alignment_utils import (
    RingAlignment,
    SegmentRings,
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
            pixel_size_um=1,
            magnification=1,
            thresh=(50, 99),
            bead_distance_um=10,
            cross_size_um=numpy.sum(numpy.max(cross, axis=0).flatten()),
            ring_radius_um=3,
        ).run()

        GT_props = regionprops(label(numpy.max(synthetic_GT, axis=0)))
        seg_props = regionprops(label(seg_rings))

        GT_centroids = {}
        seg_centroids = {}

        for i, obj in enumerate(GT_props):
            GT_centroids[i] = obj.centroid
        for i, obj in enumerate(seg_props):
            seg_centroids[i] = obj.centroid

        coor_dict = RingAlignment(
            numpy.zeros((1)),
            numpy.zeros((1)),
            pandas.DataFrame(),
            0,
            numpy.zeros((1)),
            numpy.zeros((1)),
            pandas.DataFrame(),
            0,
        ).assign_ref_to_mov(GT_centroids, seg_centroids)

        square_error = []
        for GT_coor, seg_coor in coor_dict.items():

            square_error.append(
                math.sqrt(
                    (GT_coor[0] - seg_coor[0]) ** 2 + (GT_coor[1] - seg_coor[1]) ** 2
                )
            )

        mean_square_error = mean(square_error)

        # Assert
        # Due to inferent challenges in segmenting images, the ring alignment
        # will never be pixel perfect, but this test asserts that the average
        # deviation between ground truth and result is less than 1 pixel
        assert mean_square_error < 1
