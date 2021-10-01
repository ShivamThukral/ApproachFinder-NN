# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    scene_point_clouds = end_points['scene_point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]


    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_xyz' in end_points:
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
        vote_xyz = end_points['vote_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()

    spotness_scores = end_points['spotness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_weights_per_heading_scores = end_points['weights_per_heading_scores'].detach().cpu().numpy()

    # OTHERS
    pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        scene_pc = scene_point_clouds[i,:,:]

        spotness_prob = softmax(spotness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(scene_pc, os.path.join(dump_dir, '%06d_scene_pc.ply' % (idx_beg + i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_xyz' in end_points:
            pc_util.write_ply(end_points['vote_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))

        if np.sum(spotness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,spotness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_threshold_proposal_pc.ply'%(idx_beg+i)))

        inds = (pred_mask[i,:] == 1)
        pc_util.write_ply(pred_center[i,inds , 0:3],os.path.join(dump_dir, '%06d_pred_mask_proposal_pc.ply' % (idx_beg + i)))

        #dump predicted parking spots
        if np.sum(spotness_prob > DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            parking_spots = []
            for j in range(num_proposal):
                parking_spot = np.concatenate((pred_center[i,j,0:3],pred_weights_per_heading_scores[i,j,0:config.num_heading_bin] ), axis=None)
                parking_spots.append(parking_spot)
            if len(parking_spots)>0:
                 parking_spots = np.vstack(tuple(parking_spots)) # (num_proposal, 3 + 12)
            pc_util.write_parking_ply(parking_spots, filename= os.path.join(dump_dir, '%06d_parking_spot_pc.txt'%(idx_beg+i)))

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['parking_label_mask'].cpu().numpy() # B,K2
    gt_weight_per_heading = end_points['weights_per_heading_label'].cpu().numpy() # (B, MAX_NUM_OBJ,12)

    spotness_label = end_points['spotness_label'].detach().cpu().numpy() # (B,K,)
    spotness_mask = end_points['spotness_mask'].detach().cpu().numpy() # (B,K,)

    for i in range(batch_size):
        if np.sum(spotness_label[i,:])>0:
            pc_util.write_ply(pred_center[i,spotness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(spotness_mask[i,:])>0:
            pc_util.write_ply(pred_center[i,spotness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
        inds = (gt_mask[i,:] == 1)
        pc_util.write_ply(gt_center[i,inds,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        pc_util.write_ply_color(pred_center[i,:,0:3], spotness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))


    # OPTIONAL, also dump prediction and gt details
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(" "+",".join([str(x) for x in list(t[2].flatten())]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                #fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[0].flatten())]))
                fout.write(" "+",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()
