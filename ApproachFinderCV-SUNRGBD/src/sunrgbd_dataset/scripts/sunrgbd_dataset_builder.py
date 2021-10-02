#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
import json
import open3d as o3d
from plyfile import PlyData, PlyElement

from dataset_generation import GenerateData, sunrgbd_detection_dataset, cv_pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(ROOT_DIR)
PI = 3.14159
MAX_NUM_PARKING = 200
VOTE_DIST = 1.0
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

class sunrgbdDatasetBuilder(object):
    def __init__(self, generate_data, split = 'train'):
        self.generate_data = generate_data
        self.split = split
        self.indx = 1

    def closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        indx = np.argmin(dist_2)
        return indx, dist_2[indx]

    def generateVote(self, object_pc, center):
        N = object_pc.shape[0]
        point_vote = np.zeros((N,4)) #first is the mask and rest is closest parking spot
        for i in range(N):
            indx, dist = self.closest_node(object_pc[i,:][0:3],center)
            if dist <= VOTE_DIST:
                point_vote[i,0] = 1
            point_vote[i,1:] = center[indx,:] - object_pc[i,0:3]
        return point_vote

    def viz_votes(self, object_pc, center, point_votes, scene_pc):
        points = np.vstack((object_pc[:,0:3], center))
        color_map = np.array([[1,0.706,0],[1,0,0],[0,0,0.704]])
        pcd_color = np.tile(color_map[0,:],(object_pc.shape[0],1))
        selected_color = np.tile(point_votes[:,0],(3,1)).transpose() * color_map[2,:]
        pcd_color = np.add(pcd_color, selected_color)
        parking_color = np.tile(color_map[1,:],(center.shape[0],1))
        colors = np.vstack((pcd_color,parking_color))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
        #visualise the votes
        # vote_point = object_pc[:,0:3] + point_votes[:,1:]
        # vote_point = np.vstack((vote_point,center))
        # pcd_votes = o3d.geometry.PointCloud()
        # pcd_votes.points = o3d.utility.Vector3dVector(vote_point)
        # o3d.visualization.draw_geometries([pcd_votes])
        real_pcd = o3d.geometry.PointCloud()
        real_pcd.points = o3d.utility.Vector3dVector(scene_pc[:,0:3])
        real_pcd.colors = o3d.utility.Vector3dVector(scene_pc[:,3:6] + MEAN_COLOR_RGB)
        o3d.visualization.draw_geometries([pcd + real_pcd])

    def __len__(self):
        return len(self.generate_data)

    def write_data(self, dump_dict, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        log_dict = {}
        log_dict['scene_idx'] = int(dump_dict['scene_idx'])
        log_dict['scene_name'] = int(dump_dict['scene_name'])
        #data_idx = log_dict['scene_name']*100 + dump_dict['indx']
        data_idx = dump_dict['indx']
        np.savez_compressed(os.path.join(output_folder,'%09d_scene_pc.npz'%(data_idx)),
                            scene_pc=dump_dict['scene_point_cloud'])
        np.savez_compressed(os.path.join(output_folder,'%09d_pc.npz'%(data_idx)),
                             pc=dump_dict['object_point_cloud'])
        np.savez_compressed(os.path.join(output_folder,'%09d_votes.npz'%(data_idx)),
                            point_votes=dump_dict['point_votes'])
        np.savez_compressed(os.path.join(output_folder,'%09d_parking_center.npz'%(data_idx)),
                            parking_center=dump_dict['center_label'])
        np.savez_compressed(os.path.join(output_folder,'%09d_heading_weight.npz'%(data_idx)),
                            heading_weight=dump_dict['heading_weight'])
        np.save(os.path.join(output_folder, '%09d_mask.npy' % (data_idx)), dump_dict['center_mask'])
        np.save(os.path.join(output_folder, '%09d_weights.npy' % (data_idx)), dump_dict['weights'])
        np.save(os.path.join(output_folder, '%09d_theta.npy' %(data_idx)), dump_dict['theta'])

        # generate a log file to store additional information about the scene
        log_dict = {}
        log_dict['scene_idx'] = int(dump_dict['scene_idx'])
        log_dict['scene_name'] = int(dump_dict['scene_name'])
        log_dict['global_idx'] = dump_dict['indx']
        log_dict['object_idx'] = dump_dict['object_idx']
        log_dict['class'] = dump_dict['semantic_class']
        log_dict['num_parking_spots'] = dump_dict['num_parking_spots']
        log_path = os.path.join(output_folder,'%09d_log.json'%(data_idx))
        with open(log_path,"w") as outfile:
            json.dump(log_dict, outfile)
        return data_idx

    def readData(self, output_folder, data_idx):
        #read the dataset
        scene_pc = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_scene_pc.npz')['scene_pc']
        point_cloud = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_pc.npz')['pc']
        point_votes = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_votes.npz')['point_votes']
        parking_centers = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_parking_center.npz')['parking_center']
        heading_weight = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_heading_weight.npz')['heading_weight']
        parking_mask = np.load(os.path.join(output_folder,'%09d'%(data_idx)) + '_mask.npy')
        log_path = os.path.join(output_folder,'%09d_log.json'%(data_idx))
        with open(log_path) as infile:
            log = json.load(infile)
        print('scene_pc = {}'.format(scene_pc.shape))
        print('object_pc = {}'.format(point_cloud.shape))
        print('point_votes = {}'.format(point_votes.shape))
        print('parking_centers = {}'.format(parking_centers.shape))
        print('heading_weight = {}'.format(heading_weight.shape))
        print('parking_mask = {}'.format(parking_mask.shape))
        print('Number of parking spots = {}'.format(np.sum(parking_mask)))


    def dumpData(self, output_folder, dump_dict, idx):
        dump_dir = os.path.join(output_folder,'DUMP')
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        write_ply(dump_dict['scene_point_cloud'], os.path.join(dump_dir, '%03dpc.ply'%(idx)))
        write_ply(dump_dict['object_point_cloud'], os.path.join(dump_dir, '%03d_obj.ply'%(idx)))

    def generate(self, idx):
        dump_dict = {}
        data_dict = self.generate_data.generateLabel(idx)
        # object is not in white list or no parking spots found around object
        if len(data_dict) == 0 or len(data_dict['object_pc']) == 0:
            print('Nothing found at indx location = {}'.format(idx))
            return

        dump_dict['point_clouds'] = data_dict['pc']
        object_pc = data_dict['object_pc']
        parking_data_dict = data_dict['parking_data']
        assert(parking_data_dict.keys() == object_pc.keys()) #both the keys should match
        for key in parking_data_dict:
            result = parking_data_dict[key]
            parking3d_centers = np.zeros((MAX_NUM_PARKING,3))
            heading_weight_label = np.zeros((MAX_NUM_PARKING,self.generate_data.cost_map_3d.heading_bin))
            parking_label_mask = np.zeros((MAX_NUM_PARKING))
            parking_weights = np.zeros((MAX_NUM_PARKING))
            parking_theta = np.zeros((MAX_NUM_PARKING))
            center = result['parking_locations']
            weights = result['parking_weights']
            theta = result['parking_theta']
            heading_weights = result['parking_weight_heading']
            parking_label_mask[0:center.shape[0]] = 1
            for i in range(center.shape[0]):
                parking3d_centers[i,:] = center[i,:]
                heading_weight_label[i,:] = heading_weights[i,:]
                parking_theta[i] = theta[i]
                parking_weights[i] = weights[i]
            #generate the votes
            point_votes = self.generateVote(object_pc[key],center)
            #self.viz_votes(object_pc[key], center, point_votes, data_dict['pc'])
            dump_dict['point_votes'] = point_votes
            dump_dict['object_point_cloud'] = object_pc[key]
            dump_dict['center_label'] = parking3d_centers
            dump_dict['center_mask'] = parking_label_mask
            dump_dict['heading_weight'] = heading_weight_label
            dump_dict['indx'] = self.indx
            dump_dict['scene_idx'] = data_dict['scene_idx']
            dump_dict['scene_name'] = data_dict['scene_name']
            dump_dict['object_idx'] = key
            dump_dict['semantic_class'] = data_dict['sem_cls'][key]
            dump_dict['num_parking_spots'] = np.sum(parking_label_mask)
            dump_dict['scene_point_cloud'] = data_dict['pc']
            dump_dict['weights'] = weights
            dump_dict['theta'] = theta
            self.indx += 1
            output_folder = os.path.join(ROOT_DIR, 'SUNRGBD_%s/' %(self.split))
            idx = self.write_data(dump_dict, output_folder=output_folder)
            self.readData(output_folder,idx)
            #self.dumpData(output_folder,dump_dict,idx)

if __name__ == '__main__':
    #splits = ['train']
    splits = ['train','val']
    for split in splits:
        dataset = sunrgbd_detection_dataset.SunrgbdDetectionVotesDataset(split_set=split, use_height=True, use_color=True, use_v1=False,
                                                                         augment=False, num_points=80000)
        parkingCV = cv_pipeline.ParkingSpotsCV(counter=1, dataset=dataset)
        GRID_DICT = {'XY_spread':0.9, 'theta_spread':PI/18.0,
                 'heading_bin':12, 'map_spread':4.0, 'XY_resolution':0.05}
        generate_data = GenerateData(dataset=dataset, parkingCV=parkingCV, config = GRID_DICT)
        #generate_data.generateLabel(7)    # 7, 12, 15, 16, 19
        dataset_builder = sunrgbdDatasetBuilder(generate_data, split=split)
        N = len(dataset_builder)
        print('Dataset {} Length = {}'.format(split,N))
        #dataset_builder.generate(0)
        #dataset_builder.generate(7)
        for i in range(N):
            dataset_builder.generate(i)