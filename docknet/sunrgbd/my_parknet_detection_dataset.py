# coding: utf-8

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).
Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Author: Shivam Thukral
Date: 2021

"""
import os
import sys
import numpy as np
import json
from torch.utils.data import Dataset
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import my_sunrgbd_utils as sunrgbd_utils
from my_model_util_sunrgbd import SunrgbdDatasetConfig
from my_cv_utils import O3DVisualiser, ParknetDatasetVisualiser, ObjectDumper

DC = SunrgbdDatasetConfig()  # dataset specific config
MAX_NUM_OBJ = 200  # maximum number of parking allowed per scene
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATASET_DIR = "../../ApproachFinderCV-SUNRGBD/src/sunrgbd_generation"   #sorry for hardcoding the path


class ParknetDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train',  num_points=2000,
                 use_color=False, use_height=False,
                 augment=False, scan_idx_list=None):
        assert (num_points >= 2000)
        self.data_path = os.path.join(DATASET_DIR, 'SUNRGBD_%s' % (split_set))
        #self.data_path = os.path.join(DATASET_DIR, 'DATASET_%s' % (split_set))
        self.scan_names = sorted(list(set([os.path.basename(x)[0:9] \
                                           for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.num_points = num_points
        self.split_set = split_set

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            object_point_clouds: (N,3+C)     #ST: C is the RGB and/or height
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            vote_label: (N,3) with votes XYZ
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is close to a parking spot
            park_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique parking spot
            heading_weight: (MAX_NUM_OBJ, NUM_HEADING) weight for each heading location
            sem_cls_label: (1,) semantic class of the object
            scan_idx: int scan index in scan_names list
        """
        scan_name = self.scan_names[idx]
        object_point_cloud = np.load(os.path.join(self.data_path, scan_name) + '_pc.npz')['pc']  # Nx3+C
        scene_point_cloud = np.load(os.path.join(self.data_path, scan_name) + '_scene_pc.npz')['scene_pc']  # Nx3+C
        point_votes = np.load(os.path.join(self.data_path, scan_name) + '_votes.npz')['point_votes']  # Nx4
        parking_center_label = np.load(os.path.join(self.data_path, scan_name) + '_parking_center.npz')['parking_center'] # 200, 3
        heading_weight_label = np.load(os.path.join(self.data_path, scan_name) + '_heading_weight.npz')['heading_weight'] # 200, 12
        parking_mask_label = np.load(os.path.join(self.data_path, scan_name) + '_mask.npy')     # 200,
        #weights = np.load(os.path.join(self.data_path, scan_name) + '_weights.npy')
        #theta = np.load(os.path.join(self.data_path, scan_name) + '_theta.npy')
        log_path = os.path.join(self.data_path, scan_name) + '_log.json'
        log_label = {}
        with open(log_path) as infile:
            log_label = json.load(infile)
        # ------------------------------- DATA FEATURES ------------------------------
        point_cloud = object_point_cloud
        if not self.use_color:
             point_cloud = object_point_cloud[:, 0:3]
        else:
             point_cloud = object_point_cloud[:, 0:6]           # color is already subtracted
             #point_cloud[:, 3:] = (object_point_cloud[:, 3:] - MEAN_COLOR_RGB)

        if self.use_height:
             height = object_point_cloud[:,-1]                  # last column is the height
             point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                scene_point_cloud[:,0] = -1 * scene_point_cloud[:,0]
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                point_votes[:,[1]] = -1 * point_votes[:,[1]]
                parking_center_label[:,0] = -1 * parking_center_label[:,0]
                #point_votes[:, [1, 4, 7]] = -1 * point_votes[:, [1, 4, 7]]

            # Rotation along up-axis/Z-axis
            #rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 or +30 degree
            sign = [-1,0,1][np.random.randint(0,3)]
            rot_angle = (sign * 2.0 * np.pi) / DC.num_heading_bin
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:, 1:4] = np.dot(point_cloud[:, 0:3] + point_votes[:, 1:4], np.transpose(rot_mat))
            #point_votes_end[:, 4:7] = np.dot(point_cloud[:, 0:3] + point_votes[:, 4:7], np.transpose(rot_mat))
            #point_votes_end[:, 7:10] = np.dot(point_cloud[:, 0:3] + point_votes[:, 7:10], np.transpose(rot_mat))

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            scene_point_cloud[:,0:3] = np.dot(scene_point_cloud[:, 0:3], np.transpose(rot_mat))

            heading_weight_label = np.roll(heading_weight_label, sign, axis=1)
            #theta = theta + rot_angle
            parking_center_label[:, 0:3] = np.dot(parking_center_label[:, 0:3], np.transpose(rot_mat))
            point_votes[:, 1:4] = point_votes_end[:, 1:4] - point_cloud[:, 0:3]
            #point_votes[:, 4:7] = point_votes_end[:, 4:7] - point_cloud[:, 0:3]
            #point_votes[:, 7:10] = point_votes_end[:, 7:10] - point_cloud[:, 0:3]

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= (1 + 0.4 * np.random.random(3) - 0.2)  # brightness change for each channel
                rgb_color += (0.1 * np.random.random(3) - 0.05)  # color shift for each channel
                rgb_color += np.expand_dims((0.05 * np.random.random(point_cloud.shape[0]) - 0.025),
                                            -1)  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0]) > 0.3, -1)
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            scene_point_cloud[:, 0:3] *= scale_ratio
            parking_center_label[:, 0:3] *= scale_ratio
            point_votes[:, 1:4] *= scale_ratio
            #point_votes[:, 4:7] *= scale_ratio
            #point_votes[:, 7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

        # ------------------------------- LABELS ------------------------------
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices, 0]
        point_votes = point_votes[choices, 1:]

        # # # mask for eval gt parking
        # inds = (point_votes_mask == 1)
        # pc_obj = point_cloud[inds, 0:3]
        # pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
        # pc_obj_voted1 = np.unique(pc_obj_voted1, axis=0)
        # vote_parking = np.zeros_like(parking_mask_label)
        # for i,center in enumerate(parking_center_label):
        #     if parking_mask_label[i] == 0: continue
        #     dist_2 = np.sum((pc_obj_voted1 - center) ** 2, axis=1)
        #     indx = np.argmin(dist_2)
        #     if dist_2[indx] < 0.00001:
        #         vote_parking[i] = 1

        ret_dict = {}
        ret_dict['scene_point_clouds'] = scene_point_cloud.astype(np.float32)
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)                           #ST: pc subsampled  num_points,3 (+C: RGB + Height)
        ret_dict['center_label'] = parking_center_label.astype(np.float32)[:, 0:3]          #ST: bbox center K,3
        ret_dict['weights_per_heading_label'] = heading_weight_label.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)                             #ST: num_points,3
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)                     #ST: num_points
        ret_dict['parking_label_mask'] = parking_mask_label.astype(np.float32)
        #ret_dict['weights'] = weights.astype(np.float32)
        #ret_dict['theta'] = theta.astype(np.float32)
        #ret_dict['gt_parking_voted_mask'] = vote_parking.astype(np.int64)                   # ST: this mask stores the gt parking spots which were actually voted
        # meta-data information
        #ret_dict['sem_cls_label'] = np.int64(sunrgbd_utils.type2class[log_label['class']])
        #ret_dict['scene_idx'] = np.int64(log_label['scene_name'])

        return ret_dict


def viz_votes(scene_pc, pc, point_votes, point_votes_mask, dataloader_dump_dir, visualise=False):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask == 1)
    pc_obj = pc[inds, 0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
    pc_obj_vote = pc[:,0:3] + point_votes[:,0:3]
    #pc_obj_voted2 = pc_obj + point_votes[inds, 3:6]
    #pc_obj_voted3 = pc_obj + point_votes[inds, 6:9]
    pc_util.write_ply(pc_obj,os.path.join(dataloader_dump_dir, 'pc_obj.ply'))
    #print("len = {} {}".format(pc_obj.shape, pc_obj_voted1.shape))
    #print("{}".format(pc_obj_voted1))
    pc_util.write_ply(pc_obj_voted1,  os.path.join(dataloader_dump_dir, 'pc_obj_voted1.ply'))
    #pc_util.write_ply(pc_obj_voted2, os.path.join(dataloader_dump_dir, 'pc_obj_voted2.ply'))
    #pc_util.write_ply(pc_obj_voted3, os.path.join(dataloader_dump_dir, 'pc_obj_voted3.ply'))
    pc_util.write_ply(scene_pc, os.path.join(dataloader_dump_dir, 'scene.ply'))
    if visualise:
        real_pcd = o3d.geometry.PointCloud()
        real_pcd.points = o3d.utility.Vector3dVector(pc[:, 0:3])
        real_pcd.paint_uniform_color([0.9,0.0, 0.0])
        seed_pcd = o3d.geometry.PointCloud()
        seed_pcd.points = o3d.utility.Vector3dVector(pc[inds, 0:3])
        seed_pcd.paint_uniform_color([0.0,0.9,0.0])
        parking_pcd = o3d.geometry.PointCloud()
        parking_pcd.points = o3d.utility.Vector3dVector(pc_obj_voted1)
        parking_pcd.paint_uniform_color([0.0, 0.5, 0.5])
        all_parking_pcd = o3d.geometry.PointCloud()
        all_parking_pcd.points = o3d.utility.Vector3dVector(pc_obj_vote)
        all_parking_pcd.paint_uniform_color([0.9, 0.0, 0.0])
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pc[:, 0:3])
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_pc[:, 3:6])
        o3d.visualization.draw_geometries([parking_pcd + all_parking_pcd + seed_pcd + real_pcd+ scene_pcd])




def get_sem_cls_statistics(d):
    """ Compute number of objects for each semantic class """
    #d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=False, augment=False)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i % 500 == 0: print(i)
        sample = d[i]
        class_type = sample['sem_cls_label']
        #class_type = sunrgbd_utils.type2class[sem_cls]
        if class_type not in sem_cls_cnt:
            sem_cls_cnt[class_type] = 0
        sem_cls_cnt[class_type] += 1
    print(sem_cls_cnt)
    print(sunrgbd_utils.type2class)

def get_num_parking_proposals(d):
    num_parking_locs = []
    for i in range(len(d)):
        sample = d[i]
        num_parking = np.sum(sample['parking_label_mask']).astype(np.int64)
        num_parking_locs.append(num_parking)
    num_parking_locs = np.array(num_parking_locs)
    print(np.sort(num_parking_locs))
    print("min = ",np.min(num_parking_locs))
    return np.percentile(num_parking_locs, 90)

if __name__ == '__main__':
    d_train = ParknetDetectionVotesDataset(split_set='train', num_points=2000, use_height=True, use_color=True, augment=False)
    d_val = ParknetDetectionVotesDataset(split_set='val', num_points=2000, use_height=True, use_color=True, augment=False)
    print("Train Dataset Length = {}".format(len(d_train)))
    print("Val Dataset Length = {}".format(len(d_val)))

    #get_sem_cls_statistics(d_train)
    #get_sem_cls_statistics(d_val)

    #proposals = get_num_parking_proposals(d_train)
    #print('Parknet Proposals = {}'.format(proposals))

    #TODO : visualise gt parking spots and voted as well.
    sample = d_val[12]
    # dataloader_dump_dir = os.path.join(BASE_DIR, 'data_loader_parknet_dump')
    # if not os.path.exists(dataloader_dump_dir):
    #     os.mkdir(dataloader_dump_dir)
    # viz_votes(sample['scene_point_clouds'],sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'], dataloader_dump_dir, visualise=True)

    #TODO : Generate Sample examples for training and validation
    # output_dir = os.path.join(ROOT_DIR,"demo_files/sample_scenes")
    # obj_dump = ObjectDumper(d_train, output_dir)
    # obj_dump.dumpData([0,1,2,3,40,82,84,85,175,176,177,178,213,214,217,220,221,222,241,243,245,246,261,280,267,287,293])
    # obj_dump.dumpData(
    #     [7, 12, 14, 18, 30, 52, 67, 115, 123, 129, 166, 203, 214, 216, 272, 303, 311, 312, 314, 315])  #validation


    # visualiser = ParknetDatasetVisualiser(d_val)
    # visualiser.visualise()

    # visualiser = O3DVisualiser()
    # visualiser.addPointCloud(sample['point_clouds'], 'object_pc')
    # inds = sample['parking_label_mask'] == 1
    # visualiser.addPointCloud(sample['center_label'][inds,0:3], "parking_spot")
    # inds = sample['vote_label_mask'] == 1
    # pc_obj = sample['point_clouds'][inds, 0:3]
    # pc_obj_voted = pc_obj + sample['vote_label'][inds, 0:3]
    # visualiser.addPointCloud(sample['point_clouds'][inds,0:3], "votes")
    # visualiser.visualise()



