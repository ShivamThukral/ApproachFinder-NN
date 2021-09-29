""" Deep hough voting network for 3D parking location detection in point clouds.

Author: Shivam Thukral
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from my_proposal_module import ProposalModule
from dump_helper import dump_results
from my_loss_helper import get_loss
from ap_helper import my_parse_predictions, parse_groundtruths


class ParkNet(nn.Module):
    r"""
        A deep neural network for 3D parking detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
                Number of semantics classes use to create the dataset (table, chair and toilet)
        num_heading_bin: int
        num_weight_bin: int -- un-used
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D location with its weight heading
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_weight_bin,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_weight_bin = num_weight_bin
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)  # ST: 256 is seed feature dim

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class=num_class, num_heading_bin=num_heading_bin,
                                   num_weight_bin=num_weight_bin, num_proposal=num_proposal, sampling=sampling, seed_feat_dim=256)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]
        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        end_points = self.pnet(xyz, features, end_points)
        return end_points


if __name__ == '__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from my_parknet_detection_dataset import ParknetDetectionVotesDataset, DC
    from my_loss_helper import get_loss

    NUM_POINTS = 2000 #(say)
    MAX_SPOTS = 128
    # Define model
    model = ParkNet(num_class=DC.num_class, num_heading_bin=DC.num_heading_bin, num_weight_bin=DC.num_weight_bin, input_feature_dim=1, num_proposal=MAX_SPOTS, vote_factor=1).cuda()
    try:
        # Define dataset - NOTE: I have removed number of points from here
        TRAIN_DATASET = ParknetDetectionVotesDataset(split_set='train', num_points= 2000, use_height=True, use_color=False, augment=True)
        # Model forward pass
        sample = TRAIN_DATASET[0]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        sample = {}
        inputs = {'point_clouds': torch.rand((2,NUM_POINTS, 4)).cuda()}
        # inputs = {'point_clouds': torch.rand((NUM_POINTS, 3)).unsqueeze(0).cuda()}
        # ST: generate random ground truth value as well
        sample['center_label'] = np.random.randn(MAX_SPOTS,3)
        mask = np.random.choice([0, 1], size=NUM_POINTS*2, p=[.5, .5])
        sample['vote_label_mask'] = mask
        sample['vote_label'] = np.random.randn(NUM_POINTS*2,3)
        sample['parking_label_mask'] = np.random.choice([0, 1], size=MAX_SPOTS, p=[.5, .5])
        sample['weights_per_heading_label'] = np.random.randn(MAX_SPOTS, DC.num_heading_bin)


    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key].shape)
    print('-'*30)

    # add the ground truth values
    #end_points['vote_label_mask'] = torch.from_numpy(sample['vote_label_mask']).unsqueeze(0).cuda()
    #end_points['vote_label'] = torch.from_numpy(sample['vote_label_mask']).unsqueeze(0).cuda()


    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1, 128)) #num of proposals
        #parse the predictions
        # Used for AP calculation
        CONFIG_DICT = {'remove_empty_box': True, 'use_3d_nms': True,
                       'nms_iou': 0.5,
                       'use_old_type_nms': False, 'cls_nms':False,
                       'per_class_proposal': False,
                       'conf_thresh': 0.05, 'dataset_config': DC,
                       'remove_colliding_placements': True, 'outlier_removal': True}
        # batch_pred_map_cls = my_parse_predictions(end_points, CONFIG_DICT)
        # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        # dump_results(end_points, 'tmp', DC, inference_switch=False)
        # for key in end_points:
        #     if 'loss' in key or 'acc' in key or 'ratio' in key:
        #         print(key, end_points[key].item())
        # print('-' * 30)
        # #print(end_points['batch_gt_map_cls'])
    except:
        print('Dataset has not been prepared. Skip loss and dump.')
