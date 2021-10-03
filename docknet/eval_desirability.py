# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multiprocessing import Pool, current_process
from scipy.stats import multivariate_normal
import open3d as o3d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, my_parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='parknet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=2000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=128, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps',
                    help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.05,
                    help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
FLAGS = parser.parse_args()


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
PI = 3.14159
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert (CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    DUMP_FOUT.write(out_str + '\n')
    DUMP_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from my_parknet_detection_dataset import ParknetDetectionVotesDataset, MAX_NUM_OBJ
    from my_model_util_sunrgbd import SunrgbdDatasetConfig

    DATASET_CONFIG = SunrgbdDatasetConfig()
    # Define dataset
    TEST_DATASET = ParknetDetectionVotesDataset(split_set='val', num_points=NUM_POINT, augment=False,
                                                use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)

print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=FLAGS.shuffle_dataset, num_workers=4, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model)  # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1


if FLAGS.model == 'bparknet':
    Detector = MODEL.BParkNet
else:
    Detector = MODEL.ParkNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_weight_bin = DATASET_CONFIG.num_weight_bin,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = { 'remove_colliding_placements_scene': True, 'outlier_removal': False, 'remove_colliding_placements_object' : False, 'conf_thresh':0.5,
                'min_weight_thresh': 0.01, 'dataset_config':DATASET_CONFIG}

CONFIG_DICT_COSTMAP = {'XY_spread':0.9, 'theta_spread':PI/18.0,
                 'heading_bin':12, 'map_spread':4.0, 'XY_resolution':0.05}


# ------------------------------------------------------------------------- GLOBAL CONFIG END


class CostMap3D(object):
    def __init__(self, config_dict):
        self.scalar = config_dict['XY_spread']  # How much gaussian spread we want 0.1 m or 10 cm
        self.heading_bin = config_dict['heading_bin']
        self.cov_mat = self.scalar * np.identity(3, dtype='float')
        self.cov_mat[2][2] = config_dict['theta_spread']  # 45-degrees only
        self.map_spread = config_dict['map_spread']
        self.resolution = config_dict['XY_resolution']
        self.theta_resolution = round(2.0 * PI / self.heading_bin, 5)

    def calculateGrid(self, points):
        x_bounds = np.array([0, 0], dtype='double')  # [x_min,x_max]
        y_bounds = np.array([0, 0], dtype='double')  # [y_min,y_max]
        x_bounds[0] = np.min(points[:, 0]) - self.map_spread * self.scalar
        x_bounds[1] = np.max(points[:, 0]) + self.map_spread * self.scalar
        y_bounds[0] = np.min(points[:, 1]) - self.map_spread * self.scalar
        y_bounds[1] = np.max(points[:, 1]) + self.map_spread * self.scalar
        theta_bounds = np.array([0, 2.0 * PI], dtype='float')  # [theta_min, theta_max]
        x_grid, y_grid, theta_grid = np.mgrid[x_bounds[0]:x_bounds[1]:self.resolution,
                                     y_bounds[0]:y_bounds[1]:self.resolution,
                                     theta_bounds[0]:theta_bounds[1]:self.theta_resolution]
        grid_xytheta = np.empty(x_grid.shape + (3,))
        grid_xytheta[:, :, :, 0] = x_grid
        grid_xytheta[:, :, :, 1] = y_grid
        grid_xytheta[:, :, :, 2] = theta_grid
        x_grid, y_grid = np.mgrid[x_bounds[0]:x_bounds[1]:self.resolution, y_bounds[0]:y_bounds[1]:self.resolution]
        r, c = x_grid.shape
        xyz = np.zeros((r * c, 3), dtype='float')
        xyz[:, 0] = np.reshape(x_grid, -1)
        xyz[:, 1] = np.reshape(y_grid, -1)
        return grid_xytheta, xyz

    def generateGaussian(self, actual_points, weights, theta, grid_xytheta):
        r, c, t, _ = grid_xytheta.shape
        pdf_sum = np.zeros(shape=(r, c, t), dtype='float')

        N = len(weights)
        points = np.array(actual_points)
        points[:, 2] = theta[:]  # replace the last col (height) with desired heading.
        F = [multivariate_normal(points[i], self.cov_mat) for i in range(N)]

        #results = Parallel(n_jobs=1)(delayed(F[i].pdf)(grid_xytheta) for i in range(N))
        results = [F[i].pdf(grid_xytheta) for i in range(N)]
        results = np.array(results)  # N,R,C,T                        #convert this to np array for calculations
        for i in range(N):
            pdf = np.multiply(results[i, :, :, :], weights[i])
            pdf_sum = np.add(pdf_sum, pdf)
        # normalise the whole map for 0-1 probability
        normalise = np.amax(pdf_sum[:, :, :])
        _, _, t = pdf_sum.shape
        for i in range(t):
            pdf_sum[:, :, i] = pdf_sum[:, :, i] if normalise == 0 else np.divide(pdf_sum[:, :, i], normalise)
        return pdf_sum

    def visualise(self, pdf_sum, xyz):
        _, _, t = pdf_sum.shape
        for i in range(t):
            xyz[:,2] = pdf_sum[:,:,i].flatten()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.visualization.draw_geometries([pcd])




def eval_des_scene(pred, gt, costmap3d):
    #p = current_process()
    #print('process counter:', p._identity[0], 'pid:', os.getpid())
    # common grid for evaluation
    grid_xytheta, xyz = costmap3d.calculateGrid(gt['center'])
    # heading angle
    gt_heading_class = np.argmax(gt['weights'], -1)  # num_proposal
    gt_heading_angle = gt_heading_class * ((2 * PI) / costmap3d.heading_bin)
    gt_weight = np.max(gt['weights'],-1)
    gt_pdf_sum = costmap3d.generateGaussian(gt['center'], gt_weight, gt_heading_angle, grid_xytheta)


    pred_heading_class = np.argmax(pred['weights'], -1)  # num_proposal
    pred_heading_angle = pred_heading_class * ((2 * PI) / costmap3d.heading_bin)
    pred_weight = np.max(pred['weights'], -1)
    pred_pdf_sum = costmap3d.generateGaussian(pred['center'], pred_weight, pred_heading_angle, grid_xytheta)

    mae = np.sum(np.absolute((gt_pdf_sum - pred_pdf_sum)))
    mae /= gt_pdf_sum.size      # normalise by grid-size
    return mae

def eval_des_wrapper(arguments):
    pred, gt, costmap3d = arguments
    error = eval_des_scene(pred, gt, costmap3d)
    return error

class DesirabilityError(object):
    ''' Calculating Instant Desirability Loss '''

    def __init__(self, config_dict, batch_size):
        """
        Args:
        """
        self.reset(config_dict, batch_size)

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        ret_dict = {}
        error = [] # to store all the errors fom parallel processing
        #img_ids are from 0 - len of dataset
        batch_size_count = self.batch_size
        for start_index in range(0,self.scan_cnt,batch_size_count):
            end_index = self.scan_cnt if start_index + batch_size_count > self.scan_cnt else start_index + batch_size_count
            ids = [i for i in range(start_index,end_index,1)]
            print("Eval Objects %d - %d"%(ids[0], ids[-1]))
            pred = {}  # map {id: center, heading}
            gt = {}  # map {id: gt, heading}
            for img_id in ids:
                for score, center, weight, bbox in self.pred_map_cls[img_id]:
                    if img_id not in pred: pred[img_id] = {}
                    if 'center' not in pred[img_id]:
                        pred[img_id]['center'] = []
                    if 'weights' not in pred[img_id]:
                        pred[img_id]['weights'] = []
                    pred[img_id]['obj_id'] = img_id
                    pred[img_id]['center'].append(center)
                    pred[img_id]['weights'].append(weight)

                for center, weight, bbox in self.gt_map_cls[img_id]:
                    if img_id not in gt: gt[img_id] = {}
                    if 'center' not in gt[img_id]:
                        gt[img_id]['center'] = []
                    if 'weights' not in gt[img_id]:
                        gt[img_id]['weights'] = []
                    gt[img_id]['obj_id'] = img_id
                    gt[img_id]['center'].append(center)
                    gt[img_id]['weights'].append(weight)

                if img_id in pred and len(pred[img_id]['center']) > 0:
                    pred[img_id]['center'] = np.vstack(tuple(pred[img_id]['center']))  # (num_proposal, 3)
                    pred[img_id]['weights'] = np.vstack(tuple(pred[img_id]['weights']))  # (num_proposal, 12)
                if img_id in gt and len(gt[img_id]['center']) > 0:
                    gt[img_id]['center'] = np.vstack(tuple(gt[img_id]['center']))  # (num_proposal, 3)
                    gt[img_id]['weights'] = np.vstack(tuple(gt[img_id]['weights']))  # (num_proposal, 12)
            #assert(len(gt) == len(pred))
            p = Pool(processes=batch_size_count)  # one class so one thread
            ret_values = p.map(eval_des_wrapper,
                               [(pred[img_id], gt[img_id], self.costmap3d) for img_id in gt.keys() if img_id in pred])
            p.close()
            p.join()
            error.append(ret_values)
        error = np.array(error, dtype=object)
        error = np.concatenate(error).ravel()
        #ret_dict['error_values'] = error
        ret_dict['desirability error'] = np.mean(error)
        return ret_dict

    def reset(self, config_dict, batch_size):
        self.gt_map_cls = {}  # {scan_id: [(center, heading)]}
        self.pred_map_cls = {}  # {scan_id: [(center, heading, score)]}
        self.scan_cnt = 0
        self.batch_size = batch_size
        self.costmap3d = CostMap3D(config_dict)



def evaluate_one_epoch():
    stat_dict = {}
    desirability_error = DesirabilityError(CONFIG_DICT_COSTMAP, BATCH_SIZE)
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = my_parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        desirability_error.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if batch_idx == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
    metrics_dict = desirability_error.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f' % (key, metrics_dict[key]))

    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__ == '__main__':
    print('setup done')
    eval()