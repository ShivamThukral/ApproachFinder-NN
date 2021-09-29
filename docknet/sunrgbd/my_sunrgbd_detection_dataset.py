

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Shivam Thukral
Date: 2021

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import my_sunrgbd_utils as sunrgbd_utils
from my_model_util_sunrgbd import SunrgbdDatasetConfig

DC = SunrgbdDatasetConfig()  # dataset specific config
MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1


class SunrgbdDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
                 use_color=False, use_height=False, use_v1=False,
                 augment=False, scan_idx_list=None):

        assert (num_points <= 80000)
        self.use_v1 = use_v1
        if use_v1:
            self.data_path = os.path.join(ROOT_DIR,
                                          'sunrgbd/sunrgbd_pc_bbox_votes_80k_v1_%s' % (split_set))
        else:
            self.data_path = os.path.join(ROOT_DIR,
                                          'sunrgbd/sunrgbd_pc_bbox_votes_80k_v2_%s' % (split_set))

        self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
                                           for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)     #ST: C is the RGB and/or height
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name) + '_pc.npz')['pc']  # Nx6
        bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')  # K,8 centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
        calib = np.load(os.path.join(self.data_path, scan_name) + '_calib.npy')
        img = sunrgbd_utils.load_image(os.path.join(self.data_path, scan_name) + '_img.jpg')
        d_img = sunrgbd_utils.load_depth_image(os.path.join(self.data_path, scan_name) + '_depth_img.png')

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)


        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))                #ST: L, W, H
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1               #ST: mark first K objects only used
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = bbox[3:6] * 2
            box3d_centers[i, :] = box3d_center
            box3d_sizes[i, :] = box3d_size

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
            target_bboxes[i, :] = target_bbox

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)                   #ST: pc subsampled  num_points,3 (+1: Height)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]         #ST: bbox center K,3
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9      #ST: semantic class of this object
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes                                      #ST: whole bbox are passed as well
        ret_dict['calib'] = calib.astype(np.float32)
        ret_dict['img'] = img
        ret_dict['depth_img'] = d_img
        ret_dict['scan_name'] = np.array(scan_name).astype(np.int64)
        return ret_dict

def viz_obb(pc, label, mask, bboxes ,dataloader_dump_dir =os.path.join(BASE_DIR, 'data_loader_dump')):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ == center
    mask: (K,)
    gt_max_bboxes: (K,8)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = bboxes[i]
        obb[3:6] *= 2
        obb[7] *= -1
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes,os.path.join(dataloader_dump_dir, 'gt_obbs.ply'))
    pc_util.write_ply(label[mask == 1, :], os.path.join(dataloader_dump_dir,'gt_centroids.ply'))
    pc_util.write_ply(pc, os.path.join(dataloader_dump_dir, 'pc.ply'))

def get_sem_cls_statistics(d):
    """ Compute number of objects for each semantic class """
    #d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=False, augment=False)
    sem_cls_cnt = {}
    for i in range(len(d)):
        #if i % 500 == 0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)
    print(sunrgbd_utils.type2class)


if __name__ == '__main__':
    d_train = SunrgbdDetectionVotesDataset(split_set='train', use_height=True, use_color=True, use_v1=False, augment=False, num_points=80000)
    d_val = SunrgbdDetectionVotesDataset(split_set='val', use_height=True, use_color=True, use_v1=False, augment=False, num_points=80000)
    print('Number of Samples Training : {}'.format(len(d_train)))
    print('Number of Samples Validation: {}'.format((len(d_val))))
    get_sem_cls_statistics(d_train)
    get_sem_cls_statistics(d_val)
    sample = d_train[7]       # 7, 12, 15, 16, 19
    print('Sample output keys = {}'.format(sample.keys()))
    dataloader_dump_dir = os.path.join(BASE_DIR, 'data_loader_dump')
    if not os.path.exists(dataloader_dump_dir):
        os.mkdir(dataloader_dump_dir)
    viz_obb(sample['point_clouds'][:,0:3], sample['center_label'], sample['box_label_mask'], sample['max_gt_bboxes'])
    sample['depth_img'] = np.array(sample['depth_img'])
    for key in sample:
        print(key,sample[key].shape)