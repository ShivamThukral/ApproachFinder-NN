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
import open3d.visualization.gui as gui

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)


class ParknetDataset(Dataset):
    def __init__(self, split_set='train', scan_idx_list=None):
        self.data_path = os.path.join(ROOT_DIR,
                                      'DATASET_%s' % (split_set))

        self.scan_names = sorted(list(set([os.path.basename(x)[0:9] \
                                           for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.split_set = split_set

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            object_point_clouds: (N,3+C)     #ST: C is the RGB and/or height
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            point_votes: (N,4) first column is the mask(0/1) and rest 3 are the votes
            park_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique parking spot
            heading_weight: (MAX_NUM_OBJ, NUM_HEADING) weight for each heading location
            log: meta-data about the object
            scan_name: scan name of the object
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name) + '_pc.npz')['pc']  # Nx3+C
        point_votes = np.load(os.path.join(self.data_path, scan_name) + '_votes.npz')['point_votes']  # Nx4
        parking_center_label = np.load(os.path.join(self.data_path, scan_name) + '_parking_center.npz')['parking_center'] # 100, 3
        heading_weight_label = np.load(os.path.join(self.data_path, scan_name) + '_heading_weight.npz')['heading_weight'] # 100, 12
        parking_mask_label = np.load(os.path.join(self.data_path, scan_name) + '_mask.npy')     # 100,
        log_path = os.path.join(self.data_path, scan_name) + '_log.json'
        log_label = {}
        with open(log_path) as infile:
            log_label = json.load(infile)

        # ------------------------------- LABELS ------------------------------
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)                   #ST: pc subsampled  num_points,3 (+C: RGB + Height)
        ret_dict['center_label'] = parking_center_label.astype(np.float32)         #ST: bbox center K,3
        ret_dict['heading_weight'] = heading_weight_label.astype(np.float32)
        ret_dict['point_votes'] = point_votes.astype(np.float32)                     #ST: num_points,4
        ret_dict['parking_label_mask'] = parking_mask_label.astype(np.float32)
        # meta-data information
        ret_dict['log'] = log_label
        ret_dict['scan_name'] = scan_name
        return ret_dict

    def getO3dCloud(self, points, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        if color is None:
            color = np.random.random(size=(1,3))
            pcd.paint_uniform_color(color)
        elif len(color) == 3:
            pcd.paint_uniform_color(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    def visualiseScene(self, sample, split = 'train'):
        """ Visualize point votes and point votes mask labels
        pc: (N,3 or 6), point_votes: (N,3 or 9), point_votes_mask: (N,)
        """
        point_votes = sample['point_votes']
        pc = sample['point_clouds']
        pc_color = sample['point_clouds'][:,3:6]
        center = sample['center_label']
        center_mask = sample['parking_label_mask']

        point_votes_mask = point_votes[:, 0]
        point_votes = point_votes[:, 1:]

        inds = (point_votes_mask == 1)
        seeds = pc[inds, 0:3]
        pc_obj_voted1 = seeds + point_votes[inds, :]
        inds = (center_mask == 1)
        center = center[inds,:]

        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Open3D")
        vis.show_settings = True
        object_pc = self.getO3dCloud(pc,pc_color)
        voted = self.getO3dCloud(pc_obj_voted1, [0.1, 0.7, 0.45])
        parking_spots = self.getO3dCloud(center,[0.9, 0.1, 0.1])
        seed_points = self.getO3dCloud(seeds,[0.2,0.2,0.8])
        vis.add_geometry("object_pc", object_pc)
        vis.add_geometry("parking spots", parking_spots)
        vis.add_geometry("voted_spots", voted)
        vis.add_geometry("seed_points", seed_points)

        vis.add_3d_label(parking_spots.points[0], "{}".format(sample['log']['class']))
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

    def saveSample(self, dump_dict):
        c = 'a'
        foldertype = "-"
        while c not in ['y','Y','n','N','m','M']:
            print('Enter Yes-y/No-n/Maybe-m ?')
            c = input()
            if c == 'y' or c == 'Y':
                print("Saving as Yes!")
                foldertype = 'WHITE'
            elif c == 'n' or c == 'N':
                print('Saving as No!')
                foldertype = 'BLACK'
            elif c == 'm' or c == 'M':
                print('Saving in Maybe!')
                foldertype = 'GREY'
            else:
                print("Invalid Input! Retry!")
        output_folder = os.path.join(ROOT_DIR, 'DATASET/%s/%s/' %(self.split_set, foldertype))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data_idx = int(dump_dict['scan_name'])
        np.savez_compressed(os.path.join(output_folder,'%09d_pc.npz'%(data_idx)),
                            pc=dump_dict['point_clouds'])
        np.savez_compressed(os.path.join(output_folder,'%09d_votes.npz'%(data_idx)),
                            point_votes=dump_dict['point_votes'])
        np.savez_compressed(os.path.join(output_folder,'%09d_parking_center.npz'%(data_idx)),
                            parking_center=dump_dict['center_label'])
        np.savez_compressed(os.path.join(output_folder,'%09d_heading_weight.npz'%(data_idx)),
                            heading_weight=dump_dict['heading_weight'])
        np.save(os.path.join(output_folder, '%09d_mask.npy' % (data_idx)), dump_dict['parking_label_mask'])
        log_path = os.path.join(output_folder,'%09d_log.json'%(data_idx))
        with open(log_path,"w") as outfile:
            json.dump(dump_dict['log'], outfile)
        return foldertype, data_idx

    def readData(self, folder_type, data_idx):
        output_folder = os.path.join(ROOT_DIR, 'DATASET/%s/%s/' %(self.split_set, folder_type))
        #read the dataset
        point_cloud = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_pc.npz')['pc']
        point_votes = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_votes.npz')['point_votes']
        parking_centers = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_parking_center.npz')['parking_center']
        heading_weight = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_heading_weight.npz')['heading_weight']
        parking_mask = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_mask.npy')
        log_path = os.path.join(output_folder,'%09d_log.json'%(data_idx))
        with open(log_path) as infile:
            log = json.load(infile)
        print('object_pc = {}'.format(point_cloud.shape))
        print('point_votes = {}'.format(point_votes.shape))
        print('parking_centers = {}'.format(parking_centers.shape))
        print('heading_weight = {}'.format(heading_weight.shape))
        print('parking_mask = {}'.format(parking_mask.shape))
        print('Number of parking spots = {}'.format(np.sum(parking_mask)))

    def filterDataset(self, scan_idx_list = None):
        scan_idx = [i for i in range(len(self))]
        if scan_idx_list is not None:
            scan_idx = scan_idx_list
        for i in scan_idx:
            sample = self.__getitem__(i)
            self.visualiseScene(sample, split=self.split_set)
            folder_type, data_idx = self.saveSample(sample)
            self.readData(folder_type,data_idx)
            print('--------------------------------------------------------------')

    def readDataVisualise(self, folder_type, idx):
        output_folder = os.path.join(ROOT_DIR, 'DATASET/%s/%s/' %(self.split_set, folder_type))
        file_names = sorted(list(set([os.path.basename(x)[0:9] \
                                      for x in os.listdir(output_folder)])))
        scan_name = file_names[idx]
        point_cloud = np.load(os.path.join(output_folder, scan_name) + '_pc.npz')['pc']  # Nx3+C
        point_votes = np.load(os.path.join(output_folder, scan_name) + '_votes.npz')['point_votes']  # Nx4
        parking_center_label = np.load(os.path.join(output_folder, scan_name) + '_parking_center.npz')['parking_center'] # 100, 3
        heading_weight_label = np.load(os.path.join(output_folder, scan_name) + '_heading_weight.npz')['heading_weight'] # 100, 12
        parking_mask_label = np.load(os.path.join(output_folder, scan_name) + '_mask.npy')     # 100,
        log_path = os.path.join(output_folder, scan_name) + '_log.json'
        log_label = {}
        with open(log_path) as infile:
            log_label = json.load(infile)
        # ------------------------------- LABELS ------------------------------
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)                   #ST: pc subsampled  num_points,3 (+C: RGB + Height)
        ret_dict['center_label'] = parking_center_label.astype(np.float32)         #ST: bbox center K,3
        ret_dict['heading_weight'] = heading_weight_label.astype(np.float32)
        ret_dict['point_votes'] = point_votes.astype(np.float32)                     #ST: num_points,4
        ret_dict['parking_label_mask'] = parking_mask_label.astype(np.float32)
        # meta-data information
        ret_dict['log'] = log_label
        ret_dict['scan_name'] = scan_name
        self.visualiseScene(ret_dict)

def get_sem_cls_statistics(d):
    """ Compute number of objects for each semantic class """
    #d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=False, augment=False)
    sem_cls_cnt = {}
    for i in range(len(d)):
        #if i % 500 == 0: print(i)
        sample = d[i]
        class_type = sample['log']['class']
        if class_type not in sem_cls_cnt:
            sem_cls_cnt[class_type] = 0
        sem_cls_cnt[class_type] += 1
    print(sem_cls_cnt)

def get_num_parking_proposals(d):
    num_parking_locs = []
    for i in range(len(d)):
        sample = d[i]
        num_parking = np.sum(sample['parking_label_mask']).astype(np.int8)
        num_parking_locs.append(num_parking)
    num_parking_locs = np.array(num_parking_locs,dtype=np.int8)
    return np.percentile(num_parking_locs, 90)

if __name__ == '__main__':
    print("ROOT_DIR = {}".format(ROOT_DIR))
    splits = ['train', 'val']
    for split in splits:
        dataset = ParknetDataset(split_set=split)
        print("{} Dataset Length = {}".format(split,len(dataset)))
        get_sem_cls_statistics(dataset)
        proposals = get_num_parking_proposals(dataset)
        print('Number of Proposals: {}'.format(proposals))

    dataset = ParknetDataset(split_set='train')
    #dataset.filterDataset()
    for i in range(12):
        dataset.readDataVisualise('BLACK',i)


    # d_train = ParknetDataset(split_set='train')
    # d_val = ParknetDataset(split_set='val')
    # print("Dataset Length = {}".format(len(d_train)))
    # get_sem_cls_statistics(d_train)
    # proposals = get_num_parking_proposals(d_train)
    # #list_idx = [i for i in range(73,len(d_train))]
    #d_train.filterDataset(list_idx)
    #d_val.filterDataset()
    #d_train.readDataVisualise('BLACK',2)







