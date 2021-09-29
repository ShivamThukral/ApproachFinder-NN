#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
import sys
import matplotlib.pyplot as plt


# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vcr/UBC/Research/my_votenet/parknet')
from sunrgbd import my_cv_pipeline as cv_pipeline, my_sunrgbd_detection_dataset as sunrgbd_detection_dataset
#from autorally_msgs.msg import sunrgbd_data, desiredLocation
from dataset_generation.srv import sunrgbd_data_srv, sunrgbd_data_srvRequest
from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from cost_map_3d import CostMapDataset
PI = 3.14159

class GenerateData(object):

    def __init__(self, dataset, parkingCV, config):
        self.dataset = dataset
        self.parkingCV = parkingCV
        self.cost_map_3d = CostMapDataset(config)
        rospy.wait_for_service('sunrgbd_data_groundtruth', timeout= 5)
        self.generate_label_client = rospy.ServiceProxy('sunrgbd_data_groundtruth',sunrgbd_data_srv)
        print('Generate Data Set ... ')

    def __len__(self):
        return len(self.dataset)

    def getROSSrv(self, data_dict ):
        pc, object_pc = data_dict['pc'], data_dict['object_pc']
        img, depth_img = data_dict['img'], data_dict['depth_m']
        calib, floor_height = data_dict['calib'], data_dict['floor_height']
        cam_info = CameraInfo()
        cam_info.height, cam_info.width = img.shape[0], img.shape[1]
        Rtilt, K = calib[0:3,:].flatten(), calib[3:6,:].flatten()
        cam_info.K, cam_info.R  = K, Rtilt
        obj_pc = PointCloud2()
        obj_pc = pc2.create_cloud_xyz32(obj_pc.header, object_pc[:,0:3])
        scene_pc = PointCloud2()
        scene_pc = pc2.create_cloud_xyz32(scene_pc.header, pc[:,0:3])
        bridge = CvBridge()
        ros_img = bridge.cv2_to_imgmsg(img)
        d_img = bridge.cv2_to_imgmsg(depth_img)
        data = sunrgbd_data_srvRequest()
        data.scene_pc = scene_pc
        data.depth_image = d_img
        data.image = ros_img
        data.object_pc = obj_pc
        data.calib_matrix = cam_info
        data.floor_height = floor_height
        return data

    def extractLocations(self, locations):
        N = len(locations)
        points = []
        theta = np.empty(shape=(N,1),dtype='float')
        weights = np.empty(shape=(N,1),dtype='float')
        z_pi = R.from_euler('z', 180, degrees=True) # to visualise properly
        for i in range(N):
            points.append(locations[i].location)
            heading = locations[i].heading
            qr = R.from_quat([heading.x, heading.y, heading.z, heading.w])
            qr = qr * z_pi
            z_rot = qr.as_euler('zyx',degrees=False)[0]
            if z_rot < 0:
                z_rot += 2*PI;
            theta[i] = z_rot                                                           #only consider rotation about z-axis
            weights[i] = locations[i].location_weight
        locations = np.array([[p.x,p.y,p.z] for p in points]) #unpack the point
        # assertion: each point should have one corresponding weight
        assert(locations.shape[0] == weights.shape[0])
        print('Locations Found = {}'.format(N))
        return locations, weights, theta


    def generateLabel(self, indx):
        sample = self.parkingCV[indx]
        object_pc = sample['object_pc']
        sem_cls_list = sample['sem_cls']

        if len(object_pc) == 0:
            return {}          #return empty dictionary object not in table or toilet

        data_dict = {'pc' : sample['pc'], 'depth_m' : sample['depth_m'],
                     'img' : sample['img'], 'calib' : sample['calib'],
                     'floor_height' : np.percentile(sample['pc'][:, 2], 0.99),
                     'scene_idx': sample['scan_idx'], 'scene_name': sample['scan_name']}

        # for visualisation we need to move the pcd to ground plane
        pc_viz = np.array(data_dict['pc'][:,0:3])
        #pc_viz[:,2] = np.add(pc_viz[:,2],-data_dict['floor_height'])
        #for each object pc in the scene
        parking_dict = {}
        for key in object_pc:
            try:
                print("--"*10,sample['scan_name'],"Object",key,sem_cls_list[key],"--"*10,)
                data_dict['object_pc'] = object_pc[key]  # update the object pc each time
                ros_srv = self.getROSSrv(data_dict)
                resp = self.generate_label_client(ros_srv)
                if resp.success:
                    results = {}
                    points, weight, theta = self.extractLocations(resp.desired_locations)
                    heading_weights, pdf_sum, xyz = self.cost_map_3d.calculateHeading(points, weight, theta)
                    self.cost_map_3d.publishRVIZ(pc_viz, pdf_sum, resp.desired_locations, xyz, data_dict['floor_height'])
                    results['parking_locations'] = points
                    results['parking_weights'] = weight
                    results['parking_theta'] = theta
                    results['parking_weight_heading'] = heading_weights
                    parking_dict[key] = results  #add only successful detections
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        data_dict['parking_data'] = parking_dict
        object_pc = {key: sample['object_pc'][key] for key in parking_dict}     # select only those locations where we were able to find a parking spot
        data_dict['object_pc'] = object_pc
        data_dict['pc'] = sample['pc']
        data_dict['sem_cls'] = sample['sem_cls']
        assert(parking_dict.keys() == object_pc.keys())
        return data_dict


if __name__ == '__main__':
    GRID_DICT = {'XY_spread':0.9, 'theta_spread':PI/18.0,
                 'heading_bin':12, 'map_spread':4.0, 'XY_resolution':0.05}
    dataset_train = sunrgbd_detection_dataset.SunrgbdDetectionVotesDataset(split_set='train', use_height=True, use_color=True, use_v1=False,
                                           augment=False, num_points=80000)
    dataset_val = sunrgbd_detection_dataset.SunrgbdDetectionVotesDataset(split_set='val', use_height=True, use_color=True, use_v1=False,
                                                                           augment=False, num_points=80000)
    parkingCV = cv_pipeline.ParkingSpotsCV(counter=1, dataset=dataset_train)
    generate_data = GenerateData(dataset=dataset_train, parkingCV=parkingCV, config = GRID_DICT)
    sample = generate_data.generateLabel(15)    # 7, 12, 15, 16, 19
    print(sample.keys())
    # for key in sample:
    #     if not isinstance(sample[key], dict):
    #           print(key, sample[key].shape)
    #     else:
    #          print(key, len(sample[key]))
    #
    # if len(sample) != 0:
    #     print("Parking Dict = {}".format(sample['parking_data'].keys()))
    #     print("Parking Data = {}".format(sample['parking_data'][1].keys()))
    #     print("Object Shape = {}".format([sample['object_pc'][key].shape for key in sample['object_pc']]))

