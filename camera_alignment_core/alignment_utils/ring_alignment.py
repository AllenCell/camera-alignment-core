from collections import OrderedDict
from distutils import dist
import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.spatial import distance, KDTree
from skimage import transform as tf
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment as linsum

from ..constants import LOGGER_NAME
from ..exception import AlignmentUnsuccessful
from .alignment_info import AlignmentInfo

log = logging.getLogger(LOGGER_NAME)


class RingAlignment:
    def __init__(
        self,
        ref_rings_props: pd.DataFrame,
        ref_cross_label: int,
        mov_rings_props: pd.DataFrame,
        mov_cross_label: int,
    ):
        self.ref_rings_props = ref_rings_props
        self.ref_cross_label = ref_cross_label
        self.mov_rings_props = mov_rings_props
        self.mov_cross_label = mov_cross_label

    # def assign_ref_to_mov_old(
    #     self,
    #     updated_ref_peak_dict: Dict[int, Tuple[int, int]],
    #     updated_mov_peak_dict: Dict[int, Tuple[int, int]],
    # ) -> Dict[Tuple[int, int], Tuple[int, int]]:
    #     """
    #     Assigns beads from moving image to reference image using
    #     linear_sum_assignment to reduce the distance between the same bead on
    #     the separate channels. In case where there is more beads in one channel
    #     than the other, this method will throw off the extra bead that cannot
    #     be assigned to one, single bead on the other image.

    #     :param updated_ref_peak_dict: A dictionary
    #         ({bead_number: (coor_y, coor_x)}) from reference beads
    #     :param updated_mov_peak_dict:  A dictionary
    #         ({bead_number: (coor_y, coor_x)}) from moving beads
    #     :return:
    #         ref_mov_coor_dict: A dictionary mapping the reference bead coordinates and moving bead coordinates
    #     """
    #     match_dict, weight_dict, cost_dict, max_cost = self.pos_bead_matches(updated_ref_peak_dict, updated_mov_peak_dict)
    #     edges = []
    #     weights = []
    #     costs = []
    #     positions = []
    #     total_matches = 0
    #     for idx_ref in match_dict.keys():
    #         weight_group = []
    #         edge_group = []
    #         pos_group = []
            
    #         for i, idx_mov in enumerate(match_dict[idx_ref]):
    #             edge_group.append((idx_ref, idx_mov))
    #             weight_group.append(weight_dict[idx_ref][i])
    #             pos_group.append(total_matches)
    #             costs.append(cost_dict[idx_ref][i])
    #             total_matches += 1
            
    #         edges.append((idx_ref, None))
    #         weight_group.append(0.1)
    #         costs.append(max_cost * 1.1)
    #         pos_group.append(total_matches)
    #         total_matches += 1
            
    #         weights.append(weight_group)
    #         positions.append(pos_group)
            
    #     A_eq = []
    #     for weight_group, pos_group in zip(weights, positions):
    #         row = [0] * total_matches
    #         for weight, postion in zip(weight_group, pos_group):
    #             row[postion] = weight
    #         A_eq.append(row)
                
    #     methods = ['highs','interior-point', 'simplex', 'highs-ipm']
    #     b_eq = [1]*len(A_eq)
    #     # print(np.array(A_eq).shape)
    #     for method in methods:
    #         result = linprog(
    #             c=costs,
    #             A_eq=A_eq,
    #             b_eq=b_eq,
    #             bounds=(0., 1.),
    #             method=method,
    #         )

    #         if result.success:
    #             break
    #     if not result.success:
    #         import pandas as pd
    #         pd.DataFrame(np.array(A_eq)).to_csv('A.csv')
    #         pd.DataFrame(np.array(result.x)).to_csv('x.csv')
    #         pd.DataFrame(np.array(costs)).to_csv('cost.csv')
    #         pd.DataFrame([str(e) for e in edges]).to_csv('edges.csv')
    #         raise Exception()
            
    #     # import pdb; pdb.set_trace()
            
    #     matches = []
    #     dists = []
    #     for row in A_eq:
    #         region = np.where(np.array(row)>0, result.x, np.zeros_like(result.x))
    #         index = np.argmax(region).tolist()
    #         matches.append(edges[index])
    #         dists.append(costs[index])
    #     print('max distance: ' + str(max(dists)))

    #     ref_mov_coor_dict = {}
    #     for ref_bead, mov_bead in matches:
    #         if mov_bead is None:
    #             continue
    #         ref_mov_coor_dict.update(
    #             {
    #                 updated_ref_peak_dict[ref_bead]: 
    #                     updated_mov_peak_dict[mov_bead]
    #             }
    #         )

    #     return ref_mov_coor_dict
    
    def assign_ref_to_mov(
        self,
        updated_ref_peak_dict: Dict[int, Tuple[int, int]],
        updated_mov_peak_dict: Dict[int, Tuple[int, int]],
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Assigns beads from moving image to reference image using
        linear_sum_assignment to reduce the distance between the same bead on
        the separate channels. In case where there is more beads in one channel
        than the other, this method will throw off the extra bead that cannot
        be assigned to one, single bead on the other image.

        :param updated_ref_peak_dict: A dictionary
            ({bead_number: (coor_y, coor_x)}) from reference beads
        :param updated_mov_peak_dict:  A dictionary
            ({bead_number: (coor_y, coor_x)}) from moving beads
        :return:
            ref_mov_coor_dict: A dictionary mapping the reference bead coordinates and moving bead coordinates
        """
        match_dict, weight_dict, cost_dict, max_cost = self.pos_bead_matches(updated_ref_peak_dict, updated_mov_peak_dict)
        num_ref = int(max(updated_ref_peak_dict.keys()))
        num_mov = int(max(updated_mov_peak_dict.keys()))
        C = np.ones((num_ref, num_ref + num_mov)) * (max_cost * 5)
        
        edges = []
        for ref_bead in updated_ref_peak_dict.keys():
            if ref_bead in match_dict.keys():
                for mov_bead, cost in zip(match_dict[ref_bead], cost_dict[ref_bead]):
                    C[int(ref_bead)-1, int(mov_bead)-1] = cost
            C[int(ref_bead)-1, int(ref_bead+num_mov-1)] = max_cost * 1.1
        pd.DataFrame(np.array(C)).to_csv('C.csv')
        ref_matches, mov_matches = linsum(C)
        pd.DataFrame([ref_matches, mov_matches]).to_csv('matches.csv')

        ref_mov_coor_dict = {}
        for ref_bead, mov_bead in zip(ref_matches, mov_matches):
            if mov_bead >= num_mov or C[ref_bead, mov_bead] > max_cost:
                continue
            if ref_bead+1 not in updated_ref_peak_dict.keys():
                continue
            if mov_bead+1 not in updated_mov_peak_dict.keys():
                continue
            try:
                ref_mov_coor_dict.update(
                    {
                        updated_ref_peak_dict[ref_bead+1]: 
                            updated_mov_peak_dict[mov_bead+1]
                    }
                )
            except:
                import pdb; pdb.set_trace()
        return ref_mov_coor_dict
    
    def pos_bead_matches(
        self,
        ref_peak_dict: Dict[int, Tuple[int, int]],
        mov_peak_dict: Dict[int, Tuple[int, int]],
    ) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, int], int]:
        """
        Constrain ring matching problem by identifying which rings in the 
        moving image are closer to a given reference image ring than the 
        distance between rows/columns of rings. The threshold distance is
        found by computing the median distance between rings and their 4
        nearest neighbors in the reference image.
        
        :param ref_peak_dict:  A dictionary
            ({bead_number: (coor_y, coor_x)}) from reference beads
        :param mov_peak_dict:  A dictionary
            ({bead_number: (coor_y, coor_x)}) from moving beads
        :return:
            match_dict: A dictionary mapping the reference bead numbers 
                and moving bead numbers within the threshold distance
            weight_dict: A dictionary for weight between a reference bead 
                numbers and matching moving beads for matchingS
            cost_dict: A dictionary mapping distance between a 
                reference bead numbers and matching moving beads
            thresh_dist: The threshold distance calculated
        """
        neigh = NearestNeighbors(n_neighbors=4)
        neigh.fit([coors for coors in ref_peak_dict.values()])
        mean_dist = []
        for coor in ref_peak_dict.values():
            dist, _ = neigh.kneighbors([coor], n_neighbors=4)
            mean_dist.append(dist)
        
        thresh_dist = np.median(mean_dist) * 0.8
        
                        
        tree_ref = KDTree(np.array([coors for coors in ref_peak_dict.values()]))
        tree_mov = KDTree(np.array([coors for coors in mov_peak_dict.values()]))
        neigh = tree_ref.query_ball_tree(tree_mov, thresh_dist)
        
        match_dict = {}
        weight_dict = {}
        cost_dict = {}
        ref_beads = list(ref_peak_dict.keys())
        mov_beads = list(mov_peak_dict.keys())
        for idx_ref, idxs_mov in enumerate(neigh):
            matches = []
            weights = []
            costs = []
            for idx_mov in idxs_mov:
                matches.append(ref_beads[idx_mov])
                dist = distance.euclidean(ref_peak_dict[ref_beads[idx_ref]], mov_peak_dict[mov_beads[idx_mov]])
                w = 1-dist/thresh_dist
                if w > 0.8:
                    w = 1
                # elif w < 0.2:
                #     w = 0.2
                weights.append(w)
                costs.append(dist + 0.001)
            
            match_dict[ref_beads[idx_ref]] = matches
            weight_dict[ref_beads[idx_ref]] = weights
            cost_dict[ref_beads[idx_ref]] = costs
        
        return match_dict, weight_dict, cost_dict, thresh_dist

    def rings_coor_dict(
        self, props: pd.DataFrame, cross_label: int
    ) -> Dict[int, Tuple[int, int]]:
        """
        Generate a dictionary from regionprops_table in the form of {label: (coor_y, coor_x)} for rings image
        :param props: a dataframe containing regionprops_table output
        :param cross_label: Integer value representing where the center cross is in the rings image
        :return:
            img_dict: A dictionary of label to coordinates
        """
        img_dict = {}
        for index, row in props.iterrows():
            if row["label"] is not cross_label:
                img_dict.update({row["label"]: (row["centroid-0"], row["centroid-1"])})

        return img_dict

    def change_coor_system(
        self, coor_dict: Dict[Tuple[int, int], Tuple[int, int]]
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Changes coordinates in a dictionary from {(y1, x1):(y2, x2)} to {(x1, y1): (x2, y2)}
        :param coor_dict: A dictionary of coordinates in the form of {(y1, x1):(y2, x2)}
        :return:
            An updated reversed coordinate dictionary that is {(x1, y1): (x2, y2)}
        """
        rev_yx_to_xy = {}
        for coor_ref, coor_mov in coor_dict.items():
            rev_yx_to_xy.update(
                {(coor_ref[1], coor_ref[0]): (coor_mov[1], coor_mov[0])}
            )
        return rev_yx_to_xy

    def run(
        self,
    ) -> Tuple[tf.SimilarityTransform, AlignmentInfo]:
        # get coordinate dictionaries
        ref_centroid_dict = self.rings_coor_dict(
            self.ref_rings_props, self.ref_cross_label
        )
        mov_centroid_dict = self.rings_coor_dict(
            self.mov_rings_props, self.mov_cross_label
        )

        # match reference and moving beads
        ref_mov_coor_dict = self.assign_ref_to_mov(ref_centroid_dict, mov_centroid_dict)
        # print(ref_mov_coor_dict)

        # yx to xy coordinates
        rev_coor_dict = self.change_coor_system(ref_mov_coor_dict)

        # estimate similarity transform
        tform = tf.estimate_transform(
            "similarity",
            np.asarray(list(rev_coor_dict.keys())),
            np.asarray(list(rev_coor_dict.values())),
        )

        # create alignment info
        align_info = AlignmentInfo(
            rotation=tform.rotation,
            shift_x=tform.translation[1],
            shift_y=tform.translation[0],
            z_offset=0,
            scaling=tform.scale,
        )

        return tform, align_info
