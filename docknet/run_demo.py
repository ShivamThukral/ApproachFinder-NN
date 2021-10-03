# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using ParkNet 3D parking detector to detect salient locations from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import open3d as o3d
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=2000, help='Point Number [default: 20000]')
parser.add_argument('--model_dir', default='log_docknet', help='Model path till directory')
parser.add_argument('--num_target', type=int, default=128, help='Number of proposals for Parknet [default: 128]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from pc_util import random_sampling, read_ply
from ap_helper import my_parse_predictions, flip_axis_to_camera, flip_axis_to_depth, extract_pc_in_box3d,get_3d_box, softmax, radius_outlier_removal
from my_parknet_detection_dataset import ParknetDetectionVotesDataset

def preprocess_point_cloud(point_cloud, num_points):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, num_points)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc



def plot_parking(centers, angles, origin=[0,0,0]):
    N, _ = centers.shape
    meshes = []
    for i in range(N):
        sphere_frame = o3d.geometry.TriangleMesh.create_sphere(radius = 0.025)
        sphere_frame.paint_uniform_color([0.0, 0.5, 0.5])
        sphere_frame.translate(origin + centers[i, :])
        meshes.append(sphere_frame)
        #arrow
        arrow_frame = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.008, cone_radius=0.015, cylinder_height=0.15, cone_height=0.05)
        R = arrow_frame.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        Rz = arrow_frame.get_rotation_matrix_from_xyz((0, 0, angles[i]))
        arrow_frame.rotate(R, origin)
        arrow_frame.rotate(Rz, origin)
        arrow_frame.translate(origin + centers[i, :])
        meshes.append(arrow_frame)
    return meshes

def visualise_predictions(end_points, scene):
    point_clouds = end_points['point_clouds'].cpu().numpy()
    pred_center = end_points['center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_weight = end_points['weights_per_heading_scores'].detach().cpu().detach()
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
    batch_size = point_clouds.shape[0]
    # OTHERS
    pred_mask = end_points['pred_mask']  # B,num_proposal
    pred_heading_class = np.argmax(pred_heading_weight, -1)  # B,num_proposal
    pred_heading_angle = pred_heading_class * ((2 * np.pi) / 12.0)

    for i in range(batch_size):
        scene_pc = scene
        print(scene_pc.shape)
        inds = (pred_mask[i, :] == 1)
        parking_center = pred_center[i, inds, 0:3]
        angle = pred_heading_angle[i,inds]

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pc[:, 0:3])
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_pc[:, 3:6])
        seed_pcd = o3d.geometry.PointCloud()
        seed_pcd.points = o3d.utility.Vector3dVector(seed_xyz[i,:,:])
        seed_pcd.paint_uniform_color([1,0,0])
        centers = plot_parking(parking_center, angle)
        o3d.visualization.draw_geometries(centers+[scene_pcd])



if __name__ == '__main__':
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')
    #model_dir = os.path.join(BASE_DIR, 'log_parknet')
    model_dir = os.path.join(BASE_DIR, FLAGS.model_dir)
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from my_parknet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(model_dir, 'checkpoint.tar')
        read_dir = os.path.join(demo_dir, 'sample/')
    else:
        print('Unkown dataset %s. Exiting.' % (DATASET))
        exit(-1)

    eval_config_dict = {'remove_colliding_placements_scene': True, 'remove_colliding_placements_object': True, 'outlier_removal': False,
                        'conf_thresh': 0.45, 'min_weight_thresh' : 0.1, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('parknet')  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1
    net = MODEL.ParkNet(num_proposal=FLAGS.num_target, input_feature_dim=num_input_channel, vote_factor=1,
                        sampling='vote_fps', num_class=DC.num_class,
                        num_heading_bin=DC.num_heading_bin,
                        num_weight_bin = DC.num_weight_bin,
                        ).to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    # Load and preprocess input point cloud
    net.eval()  # set model to eval mode (for bn and dp)

    #read all the files in the dir and show output
    scan_idx_list = [2,3,12,18,67,216,314]
    scan_names = sorted(list(set([os.path.basename(x) for x in os.listdir(read_dir)])))
    if scan_idx_list is not None:
        scan_names = scan_idx_list

    for scan_name in scan_names:
        scan_name = np.int64(scan_name)
        point_cloud = np.load(os.path.join(read_dir, "%04d/%04d_pc.npz"%(scan_name,scan_name)))['pc']
        scene_cloud = np.load(os.path.join(read_dir,"%04d/%04d_scene_pc.npz"%(scan_name,scan_name)))['scene_pc']
        pc = preprocess_point_cloud(point_cloud, FLAGS.num_point)
        scene_point_cloud = preprocess_point_cloud(scene_cloud, 20000)
        print('Loaded point cloud data: %04d' % (scan_name))

        # Model inference
        inputs = {'point_clouds': torch.from_numpy(pc).to(device), 'scene_point_clouds': torch.from_numpy(scene_point_cloud).to(device)}

        tic = time.time()
        with torch.no_grad():
            end_points = net(inputs)
        toc = time.time()
        print('Inference time: %f' % (toc - tic))
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['scene_point_clouds'] = inputs['scene_point_clouds']
        pred_center = end_points['center']  # B,num_proposal,3
        pred_map_cls = my_parse_predictions(end_points, eval_config_dict)
        print('Finished detection. %d object detected.' % (len(pred_map_cls[0])))


        dump_dir = os.path.join(demo_dir, '%s_results/%04d' % (FLAGS.dataset, scan_name))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        MODEL.dump_results(end_points, dump_dir, DC, True)
        print('Dumped detection results to folder %s' % (dump_dir))

        visualise_predictions(end_points, scene_cloud)
