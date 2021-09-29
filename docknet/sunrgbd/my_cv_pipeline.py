"""
Author: Shivam Thukral
File Status: Ready and Checked
"""


import os
import sys
import numpy as np
from PIL import Image
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import my_sunrgbd_utils as sunrgbd_utils
from my_sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
from my_cv_utils import project_upright_depth_to_image

NUM_POINTS = 2000 # PCD's with lesser number are removed from the training set.
WHITE_LIST = ['toilet', 'table']

class ParkingSpotsCV(object):

    def __init__(self, counter, dataset, dump_dir = os.path.join(BASE_DIR, 'data_cv_dump')):
        self.counter = counter
        self.dump_dir = dump_dir
        self.dataset = dataset
        if not os.path.exists(self.dump_dir):
            os.mkdir(self.dump_dir)

    def datasetLength(self):
        return len(self.dataset)

    def extractObjectPCD(self, sample):
        semantic_class = {}
        object_pc = {}
        pc = sample['point_clouds']
        mask = sample['box_label_mask']
        sem_cls = sample['sem_cls_label']
        bboxes = sample['max_gt_bboxes']
        for j in range(len(mask)):
            if mask[j] == 0: continue
            # Find all points in this object's OBB
            box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(bboxes[j,0:3],
                                                           bboxes[j,3:6],
                                                           bboxes[j,7])  # S: convert the BB params to 8x3 bounding box values
            pc_in_box3d, inds = sunrgbd_utils.extract_pc_in_box3d( \
                pc, box3d_pts_3d)  # S: xyz + index from N
            obj_sem_cls = sunrgbd_utils.class2type[sem_cls[j]]
            if pc_in_box3d.shape[0] >= NUM_POINTS and obj_sem_cls in WHITE_LIST:      #store only those points which are >= NUM_POINTS
                object_pc[self.counter] = pc_in_box3d
                semantic_class[self.counter] = sunrgbd_utils.class2type[sem_cls[j]]
                self.counter += 1
        return pc, object_pc, semantic_class

    def extractDepthImage(self, pc, calib, img):
        uv, d = project_upright_depth_to_image(pc, calib)
        # create the depth image
        l, w, _ = img.shape
        depth_img = np.zeros((l, w, 1), dtype=float)
        for i in range(uv.shape[0]):
            depth = d[i]
            u, v = int(np.round(uv[i, 0])), int(np.round(uv[i, 1]))
            if u < w and v < l:
                depth_img[v, u] = depth

        # import matplotlib.pyplot as plt
        # cmap = plt.cm.get_cmap('hsv', 256)
        # cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        #
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # for i in range(uv.shape[0]):
        #     depth = d[i]
        #     color = cmap[int(120.0 / depth), :]
        #     cv2.circle(img, (int(np.round(uv[i, 0])), int(np.round(uv[i, 1]))), 2,
        #                color=tuple(color), thickness=-1)
        #
        # cv2.imshow('Single Channel Window', depth_img)
        # cv2.imshow('Img', img)
        # cv2.waitKey(0)
        # print('Point UV:', uv.shape)
        # print('Point depth:', d.shape)
        return depth_img

    def convertDepthImage(self, depth_img):
        depth = np.asarray(depth_img, np.uint16)
        depthInpaint = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
        depthInpaint = depthInpaint.astype(np.single) / 1000
        depthInpaint[depthInpaint > 8] = 8
        return depthInpaint

    def __getitem__(self, item):
        cv_sample = {}
        sample = self.dataset[item]  # 7, 12, 15, 16, 19
        center_label = sample['center_label']
        pc, object_pc, semantic_class = self.extractObjectPCD(sample)
        # depth_image = self.extractDepthImage(pc[:, 0:3], sample['calib'], sample['img'])
        # cv_sample['depth_img'] = depth_image
        depth_img = sample['depth_img']
        # convert into depth value to meters
        depth_m = self.convertDepthImage(depth_img)
        # depth conversion end
        cv_sample['pc'] = pc
        cv_sample['object_pc'] = object_pc
        cv_sample['img'] = sample['img']
        cv_sample['sem_cls'] = semantic_class
        cv_sample['calib'] = sample['calib']
        cv_sample['depth_img'] = np.array(depth_img)
        cv_sample['depth_m'] = np.array(depth_m)
        cv_sample['scan_idx'] = sample['scan_idx']
        cv_sample['scan_name'] = sample['scan_name']
        return cv_sample

    def writeSample(self, sample):
        pc = sample['pc']
        pc_util.write_ply(pc, os.path.join(self.dump_dir, 'pc.ply'))
        object_pc = sample['object_pc']
        for key in object_pc:
            pc_util.write_ply(object_pc[key], os.path.join(self.dump_dir, '%06d_object.ply' % (key)))
        img = sample['img']
        Image.fromarray(img).save(os.path.join(self.dump_dir, 'img.jpg'))
        log = {'class':sample['sem_cls'],'scan_idx':int(sample['scan_idx']), 'scan_name':int(sample['scan_name'])}
        log_path = os.path.join(self.dump_dir, 'log.json')
        with open(log_path, "w") as outfile:
            json.dump(log, outfile)


def get_sem_cls_statistics(d):
    """ Compute number of objects for each semantic class """
    #d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=False, augment=False)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i % 500 == 0: print(i)
        sample = d[i]
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in range(len(mask)):
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)
    print(sunrgbd_utils.type2class)


if __name__ == '__main__':

    dataset_train = SunrgbdDetectionVotesDataset(split_set='train', use_height=True, use_color=True, use_v1=False,
                                                augment=False, num_points=80000)
    dataset_val = SunrgbdDetectionVotesDataset(split_set='val', use_height='True', use_color=True, use_v1=False,
                                               augment=False, num_points=80000)
    #---- DATASET SUMMARY ---------
    print("Train Dataset Length = {}".format(len(dataset_train)))
    print("Val Dataset Length = {}".format(len(dataset_val)))
    #get_sem_cls_statistics(dataset_train)
    #get_sem_cls_statistics(dataset_val)

    # ---- Extract and visualise one sample --------
    parkingCV = ParkingSpotsCV(counter=1, dataset=dataset_train)
    sample = parkingCV[5]   # 7, 12, 15, 16, 19
    parkingCV.writeSample(sample)
    print('ParkingCV keys = {}'.format(sample.keys()))
    # import matplotlib.pyplot as plt
    # plt.imshow(sample['depth_m'])
    # plt.show()

