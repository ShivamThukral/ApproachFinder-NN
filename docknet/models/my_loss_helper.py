# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.3 #ST0.6
NEAR_THRESHOLD = 0.15 #ST0.3
GT_VOTE_FACTOR = 1  # number of GT votes per point : ST: changed from 3 to 1 for our loss function
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        ST: if the seed point is near to any parking spot (<=0.2m) then
        votes_label_mask == 1 and requires it to vote for a parking spot.


        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1]  # B,num_seed,3 = ST: B, 1024, 3
    vote_xyz = end_points['vote_xyz']  # B,num_seed*vote_factor,3 = ST: B, 1024, 3
    seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)   # B,num_seed
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)       #B,num_seed,3 selected from the gt data
    seed_gt_votes += end_points['seed_xyz'].repeat(1, 1, GT_VOTE_FACTOR)                             # ST: this is the location where the seed points in gt should have voted
    #########################################################
    # In the above line I have changed seed_gt_votes += end_points['seed_xyz'].repeat(1, 1, 3)
    #########################################################

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1,
                                     3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR,
                                               3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
    return vote_loss

def compute_spotness_loss(end_points):
    """ Compute objectness loss for the proposals.

       Args:
           end_points: dict (read-only)

       Returns:
           spotness_loss: scalar Tensor
           spotness_label: (batch_size, num_seed) Tensor with value 0 or 1
           spotness_mask: (batch_size, num_seed) Tensor with value 0 or 1
           spot_assignment: (batch_size, num_seed) Tensor with long int
               within [0,num_gt_object-1]
       """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate spotness label and mask
    # spotness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # spotness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    spotness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    spotness_mask = torch.zeros((B, K)).cuda()
    spotness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    spotness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    spotness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    spotness_scores = end_points['spotness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    spotness_loss = criterion(spotness_scores.transpose(2, 1), spotness_label)
    spotness_loss = torch.sum(spotness_loss * spotness_mask) / (torch.sum(spotness_mask) + 1e-6)

    # Set assignment
    spot_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return spotness_loss, spotness_label, spotness_mask, spot_assignment

def compute_center_loss(end_points):
    """ Compute center loss between predicted and ground truth

        Args:
            end_points: dict (read-only)

        Returns:
            center_loss
    """
    # Compute center loss

    pred_center = end_points['center']
    gt_center = end_points['center_label'][:, :, 0:3]  # center_label: (batch,MAX_NUM_SPOTS,3) for GT spot XYZ
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    spot_label_mask = end_points['parking_label_mask']         # ST: box label mask is spot label mask
    spotness_label = end_points['spotness_label'].float()
    #print('Mask = {} , Label = {}, dist2 = {}'.format(spot_label_mask.shape, spotness_label.shape,dist2.shape))
    centroid_reg_loss1 = \
        torch.sum(dist1 * spotness_label) / (torch.sum(spotness_label) + 1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2 * spot_label_mask) / (torch.sum(spot_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    return  center_loss

def compute_weight_heading_loss(end_points,config):
    spot_assignment = end_points['spot_assignment']         # (B,K)
    spotness_label = end_points['spotness_label'].float()
    num_heading_bin = config.num_heading_bin

    heading_weight_label = torch.gather(end_points['weights_per_heading_label'], 1,
                                       spot_assignment.unsqueeze(-1).repeat(1, 1, num_heading_bin))  # select (B,K,num_bin_heading) from (B,K2,num_bin_heading)

    predicted_heading_weight = end_points['weights_per_heading_scores'] # network output - B,num_seed,num_bin_heading
    predicted_heading_weight_loss = torch.mean(
        huber_loss(predicted_heading_weight - heading_weight_label, delta=1.0),
        -1)  # (B,K,3) -> (B,K)
    heading_weight_loss = torch.sum(predicted_heading_weight_loss * spotness_label) / (
                torch.sum(spotness_label) + 1e-6)

    return heading_weight_loss



def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    spotness_loss, spotness_label, spotness_mask, spot_assignment = \
        compute_spotness_loss(end_points)
    end_points['spotness_loss'] = spotness_loss
    end_points['spotness_label'] = spotness_label
    end_points['spotness_mask'] = spotness_mask
    end_points['spot_assignment'] = spot_assignment
    total_num_proposal = spotness_label.shape[0] * spotness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(spotness_label.float().cuda()) / float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(spotness_mask.float()) / float(total_num_proposal) - end_points['pos_ratio']

    # center loss
    center_loss = compute_center_loss(end_points)
    end_points['center_loss'] = center_loss

    # # heading + weight loss
    weight_heading_reg_loss = compute_weight_heading_loss(end_points,config)
    end_points['weight_heading_loss'] = weight_heading_reg_loss

    # Final loss function
    loss = vote_loss + 0.5*spotness_loss + center_loss + weight_heading_reg_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['spotness_scores'], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == spotness_label.long()).float() * spotness_mask) / (
                torch.sum(spotness_mask) + 1e-6)
    end_points['spot_acc'] = obj_acc
    return loss, end_points
