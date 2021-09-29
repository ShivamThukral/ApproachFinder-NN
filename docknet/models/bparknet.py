import torch
import torch.nn as nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from my_proposal_module import ProposalModule
from dump_helper import dump_results
from my_loss_helper_bparknet import get_loss
from ap_helper import my_parse_predictions, parse_groundtruths

class BParkNet(nn.Module):
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

         # Box proposal, aggregation and detection
         self.pnet = ProposalModule(num_class=num_class, num_heading_bin=num_heading_bin,
                                    num_weight_bin=num_weight_bin, num_proposal=num_proposal, sampling=sampling,
                                    seed_feat_dim=256)
         print('Base Parknet loaded.....')

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
         xyz = end_points['fp2_xyz']
         features = end_points['fp2_features']
         end_points['seed_inds'] = end_points['fp2_inds']
         end_points['seed_xyz'] = xyz
         end_points['seed_features'] = features

         # Directly predict bounding boxes (skips voting)
         end_points = self.pnet(xyz, features, end_points)

         return end_points


if __name__ == '__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from my_parknet_detection_dataset import ParknetDetectionVotesDataset, DC

    NUM_POINTS = 2000  # (say)
    MAX_PROPOSALS = 128

    TRAIN_DATASET = ParknetDetectionVotesDataset(split_set='train', num_points=2000, use_height=True, use_color=False,
                                                 augment=True)
    # Define model
    model = BParkNet(num_class=DC.num_class, num_heading_bin=DC.num_heading_bin, num_weight_bin=DC.num_weight_bin, input_feature_dim=1, num_proposal=MAX_PROPOSALS, vote_factor=1).cuda()
    # Model forward pass
    sample = TRAIN_DATASET[0]
    inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    end_points = model(inputs)

    for key in end_points:
        print(key, end_points[key].shape)
    print('-'*30)

    for key in sample:
        end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
    loss, end_points = get_loss(end_points, DC)
    print('loss', loss)
    end_points['point_clouds'] = inputs['point_clouds']
    end_points['pred_mask'] = np.ones((1, MAX_PROPOSALS))
    CONFIG_DICT = { 'conf_thresh': 0.05, 'dataset_config': DC,
                   'remove_colliding_placements_scene': True, 'remove_colliding_placements_object': True,
                    'min_weight_thresh': 0.01,'outlier_removal': True}
    batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
    batch_pred_map_cls = my_parse_predictions(end_points, CONFIG_DICT)
    dump_results(end_points, 'tmp', DC, inference_switch=False)
