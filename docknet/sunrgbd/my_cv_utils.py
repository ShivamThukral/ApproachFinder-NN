import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
import open3d as o3d
import open3d.visualization.gui as gui

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[:, 1] *= -1
    return pc2

def project_upright_depth_to_camera(pc, Rtilt):
    ''' project point cloud from depth coord to camera coordinate
        Input: (N,3) Output: (N,3)
    '''
    # Project upright depth to depth coordinate
    pc2 = np.dot(np.transpose(Rtilt), np.transpose(pc[:, 0:3]))  # (3,n)
    return flip_axis_to_camera(np.transpose(pc2))

def project_upright_depth_to_image(pc, calib):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    Rtilt = calib[0:3,:]
    K = calib[3:6,:]
    pc2 = project_upright_depth_to_camera(pc, Rtilt)
    uv = np.dot(pc2, np.transpose(K))  # (n,3)
    uv[:, 0] /= uv[:, 2]
    uv[:, 1] /= uv[:, 2]
    return uv[:, 0:2], pc2[:, 2]

if __name__ == '__main__':
    print("hello")
    pc = np.asarray([[0.960995, 1.02557, -0.991231],
                     [0.981655, 1.0711, -0.991231],
                     [1.00231, 1.11663, -0.991231],
                     [1.02297, 1.16217, -0.991231],
                     [1.04363, 1.2077, -0.991231]])
    K = np.asarray([[529.5, 0, 365], [0, 529.5, 265], [0, 0, 1]])
    Rtilt = np.asarray([[0.99998, -0.001044, 0.005859],
                    [-0.001044, 0.93849, 0.34529],
                    [-0.005859, -0.34529, 0.93848]])
    calib = np.vstack((Rtilt, K))
    uv, d = project_upright_depth_to_image(pc,calib)
    print(uv)


class O3DVisualiser(object):
    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()
        self.vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        self.vis.show_settings = True



    def addPointCloud(self, pc, name, color=None):
        pcd = o3d.geometry.PointCloud()
        if color is None:
            color = np.random.random((3))
        pcd.points = o3d.utility.Vector3dVector(pc[:, 0:3])
        pcd.paint_uniform_color(color)
        self.vis.add_geometry(name, pcd)
        for idx in range(0, len(pcd.points),500):
            self.vis.add_3d_label(pcd.points[idx], "{}".format(name))
        #o3d.visualization.gui.Label3D(color, np.mean(pc[:,0:3],axis=0), name)

    def visualise(self):
        self.vis.reset_camera_to_default()
        self.app.add_window(self.vis)
        self.app.run()

class ParknetDatasetVisualiser(object):
    def __init__(self, dataset, show_metadata = False):
        self.dataset = dataset
        self.app = gui.Application.instance
        self.app.initialize()
        self.type2class = {'table': 0, 'chair': 1, 'toilet': 2}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.metadata = show_metadata


    def visualise(self, idxs=None):
        if idxs is None:
            idxs = np.array(range(0, len(self.dataset)))
            #np.random.seed(0)
            #np.random.shuffle(idxs)

        for idx in idxs:
            data_idx = idx
            print('-' * 10, 'data index: ', data_idx)
            sample = self.dataset[data_idx]
            self.showSample(sample)

    def getO3dCloud(self, points, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        if color is None:
            color = np.random.random(size=(1, 3))
            pcd.paint_uniform_color(color)
        elif len(color) == 3:
            pcd.paint_uniform_color(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    def showSample(self, sample):
        self.vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        self.vis.show_settings = True

        if 'scene_idx' in sample.keys():
            scene_id = sample['scene_idx']
            sem_cls = sample['sem_cls_label']
            print("Scene ID = {}\nSemantic Class Type = {}\nClass Name = {}".format(scene_id,sem_cls,self.class2type[sem_cls]))
            scene_pc = sample['scene_point_clouds']
            pc = self.getO3dCloud(scene_pc[:,0:3], scene_pc[:,3:6])
            self.vis.add_geometry("scene_pc", pc)

        object_pc = sample['point_clouds']
        center = sample['center_label']
        heading_weights = sample['weights_per_heading_label']
        votes = sample['vote_label']
        votes_mask = sample['vote_label_mask']
        parking_mask = sample['parking_label_mask']
        #------- Shape display -----------------
        for key in sample:
            print(key,sample[key].shape)
        print("# of Parking = {}".format(np.sum(parking_mask)))

        #-------- Visualise -------------------
        inds = (votes_mask == 1)
        seeds = object_pc[inds, 0:3]
        pc_obj_voted1 = seeds + votes[inds, :]
        inds = (parking_mask == 1)
        center = center[inds, :]
        pc_color = object_pc[:, 3:6]
        object_pc = self.getO3dCloud(object_pc, pc_color)
        voted = self.getO3dCloud(pc_obj_voted1, [0.1, 0.7, 0.45])
        parking_spots = self.getO3dCloud(center, [0.9, 0.1, 0.1])
        seed_points = self.getO3dCloud(seeds, [0.2, 0.2, 0.8])
        #o3d.visualization.draw_geometries([ object_pc + pc])
        self.vis.add_geometry("object_pc", object_pc)
        self.vis.add_geometry("parking spots", parking_spots)
        self.vis.add_geometry("voted_spots", voted)
        self.vis.add_geometry("seed_points", seed_points)
        self.vis.reset_camera_to_default()
        self.app.add_window(self.vis)
        self.app.run()
        self.app.quit()


class ObjectDumper(object):
    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = os.path.join(output_dir, self.dataset.split_set)

    def dumpData(self, indxs = None):
        if indxs is None:
            print('Enter some indices....')
            return

        for idx in indxs:
            sample = self.dataset[idx]
            #---- Dump the data
            output_folder = os.path.join(self.output_dir, "%04d"%(idx))
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            np.savez_compressed(os.path.join(output_folder, '%04d_scene_pc.npz' % (idx)),
                                scene_pc=sample['scene_point_clouds'])
            np.savez_compressed(os.path.join(output_folder, '%04d_pc.npz' % (idx)),
                                pc=sample['point_clouds'])
            print('-'*10 , "Dumped object %d"%(idx))


