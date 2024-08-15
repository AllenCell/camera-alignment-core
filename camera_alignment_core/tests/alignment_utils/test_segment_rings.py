import math
from typing import Dict, List

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

        ref_data_dict: Dict[str, List[int]] = {
            "label": [],
            "centroid-0": [],
            "centroid-1": [],
        }
        mov_data_dict: Dict[str, List[int]] = {
            "label": [],
            "centroid-0": [],
            "centroid-1": [],
        }

        # initialize to all zeros
        synthetic_GT = numpy.zeros((11, 405, 611), numpy.uint16)
        count = 0
        # insert 9 spheres of 1s within the zeros, centered at the x,y intersections, z=6
        for y in numpy.arange(10, 400, 30):
            for x in numpy.arange(10, 610, 30):
                synthetic_GT[
                    6 - ball_radius : 6 + ball_radius + 1,
                    y - ball_radius : y + ball_radius + 1,
                    x - ball_radius : x + ball_radius + 1,
                ] = ball(ball_radius)
                ref_data_dict["label"].append(count)
                ref_data_dict["centroid-0"].append(x)
                ref_data_dict["centroid-1"].append(y)
                mov_data_dict["label"].append(count)
                mov_data_dict["centroid-0"].append(x + ball_radius)
                mov_data_dict["centroid-1"].append(y + ball_radius)
                count += 1

        # compose a cross of '1' in a field of 0
        cross = numpy.zeros((11, 31, 31))
        cross[:, 13:20, :] = 1
        cross[:, :, 13:20] = 1
        cross[:2, :, :] = 0
        cross[-2:, :, :] = 0

        # add the cross to the field of zeros
        synthetic_GT[:, 174:205, 294:325] = cross
        synthetic_image = (synthetic_GT * 20000) + 5000
        # smooth out the image over the expanded pixels/array size
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
            ref_data_dict["centroid-0"][i] = obj.centroid[0]
            ref_data_dict["centroid-1"][i] = obj.centroid[1]
        for i, obj in enumerate(seg_props):
            seg_centroids[i] = obj.centroid
            mov_data_dict["centroid-0"][i] = obj.centroid[0]
            mov_data_dict["centroid-1"][i] = obj.centroid[1]

        coor_dict = RingAlignment(
            pandas.DataFrame(ref_data_dict),
            0,
            pandas.DataFrame(mov_data_dict),
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
