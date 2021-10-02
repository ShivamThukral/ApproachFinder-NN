#!/usr/bin/env python
import numpy as np
import sys
import rospy
import ros_numpy
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from scipy import spatial
from geometry_msgs.msg import Pose, Vector3, Quaternion
from rviz_tools import RvizMarkers
import open3d as o3d
from numpy_ros_pcd import convertCloudFromOpen3dToRos,convertCloudFromRosToOpen3d

PI = 3.14159

class CostMapDataset:
    def __init__(self, config):
        rospy.init_node('costmap_heading')
        self.scalar = config['XY_spread'] # How much gaussian spread we want 0.1 m or 10 cm
        self.cov_mat = self.scalar * np.identity(3, dtype = 'float')
        self.cov_mat[2][2] = config['theta_spread']   # 45-degrees only
        self.heading_bin = config['heading_bin']
        self.map_spread = config['map_spread']
        self.resolution = config['XY_resolution']
        self.theta_resolution = round(2.0*PI/self.heading_bin,5)
        self.markers = RvizMarkers('odom', '/dataset/locations_rviz')
        self.cloud_pub = rospy.Publisher('/dataset/pc', PointCloud2, queue_size=1)
        self.inst_pub = [rospy.Publisher("/dataset/instant_3d_" + str(int(i*self.theta_resolution * 180/PI)) , PointCloud2, queue_size=1) for i in range(self.heading_bin)]

    def generateGaussian(self,  actual_points, weights, theta, grid_xytheta):
        r,c,t,_ = grid_xytheta.shape
        pdf_sum = np.zeros(shape=(r, c, t),dtype='float')

        N = len(weights)
        points = np.array(actual_points)
        points[:,2] = theta[:,0] #replace the last col (height) with desired heading.
        F = [multivariate_normal(points[i], self.cov_mat) for i in range(N)]

        results = Parallel(n_jobs=4)(delayed(F[i].pdf)(grid_xytheta) for i in range(N))
        results = np.array(results)                 # N,R,C,T                        #convert this to np array for calculations
        for i in range(N):
            pdf = np.multiply(results[i,:,:,:],weights[i])
            pdf_sum = np.add(pdf_sum, pdf)
        # normalise the wholw map for 0-1 probability
        normalise = np.amax(pdf_sum[:,:,:])
        _, _, t = pdf_sum.shape
        for i in range(t):
            pdf_sum[:,:,i] = pdf_sum[:,:,i] if normalise == 0 else np.divide(pdf_sum[:,:,i], normalise)
        return pdf_sum

    def calculateGrid(self, points):
        x_bounds = np.array([0,0],dtype = 'double') # [x_min,x_max]
        y_bounds = np.array([0,0],dtype = 'double') # [y_min,y_max]
        x_bounds[0] = np.min(points[:,0]) - self.map_spread*self.scalar
        x_bounds[1] = np.max(points[:,0]) + self.map_spread*self.scalar
        y_bounds[0] = np.min(points[:,1]) - self.map_spread*self.scalar
        y_bounds[1] = np.max(points[:,1]) + self.map_spread*self.scalar
        theta_bounds = np.array([0,2.0*PI],dtype = 'float')       # [theta_min, theta_max]
        x_grid, y_grid, theta_grid = np.mgrid[x_bounds[0]:x_bounds[1]:self.resolution, y_bounds[0]:y_bounds[1]:self.resolution, theta_bounds[0]:theta_bounds[1]:self.theta_resolution]
        grid_xytheta = np.empty(x_grid.shape + (3,))
        grid_xytheta[:, :, :,0] = x_grid
        grid_xytheta[:, :, :,1] = y_grid
        grid_xytheta[:, :, :,2] = theta_grid
        x_grid, y_grid = np.mgrid[x_bounds[0]:x_bounds[1]:self.resolution, y_bounds[0]:y_bounds[1]:self.resolution]
        r,c = x_grid.shape
        xyz = np.zeros((r*c, 3),dtype='float')
        xyz[:, 0] = np.reshape(x_grid, -1)
        xyz[:, 1] = np.reshape(y_grid, -1)
        return grid_xytheta, xyz

    def findNearest(self, xyz, point):
        tree=spatial.cKDTree(xyz)
        I = tree.query(point)
        return I[1]

    def calculateHeading(self, points, weight, theta):
        grid_xytheta, xyz = self.calculateGrid(points)
        pdf_sum = self.generateGaussian(points, weight, theta, grid_xytheta)
        indx = self.findNearest(xyz[:,0:2],points[:,0:2])
        p = [Point(xyz[i][0],xyz[i][1],xyz[i][2]) for i in indx]
        self.markers.publishSpheres(p, 'blue', 0.12, 4) # path, color, diameter, lifetime
        heading_weights = []
        N = len(weight)
        for i in range(N):
            weights = []
            for j in range(self.heading_bin):
                val = pdf_sum[:,:,j].flatten()
                weights.append(val[indx[i]])
            heading_weights.append(np.asarray(weights))
        heading_weights = np.array(heading_weights)
        return heading_weights, pdf_sum, xyz


    def publishRVIZ(self, cloud, pdf_sum, locations, xyz, floor_height, RVIZ_DURATION = 4):
        #publish cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(cloud)
        ros_cloud = convertCloudFromOpen3dToRos(pcd_o3d,'odom')
        self.cloud_pub.publish(ros_cloud)
        # publish locations
        N = len(locations)
        scale = Vector3(0.5, 0.05, 0.05) # x=length, y=height, z=height # single value for length (height is relative)
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
            quat = Quaternion(qr.as_quat()[0],qr.as_quat()[1],qr.as_quat()[2],qr.as_quat()[3])
            P = Pose(points[i],quat)
            self.markers.publishArrow(P, 'yellow', scale, RVIZ_DURATION)                # pose, color, arrow_length, lifetime
            theta[i] = z_rot                                                           #only consider rotation about z-axis
            weights[i] = locations[i].location_weight
        self.markers.publishSpheres(points, 'green', 0.09, RVIZ_DURATION) # path, color, diameter, lifetime
        locations = np.array([[p.x,p.y,p.z] for p in points]) #unpack the point
        # assertion: each point should have one corresponding weight
        assert(locations.shape[0] == weights.shape[0])
        #publish instant maps
        for i in range(self.heading_bin):
            xyz[:, 2] = pdf_sum[:,:,i].flatten()
            open3d_pcd = o3d.geometry.PointCloud()
            open3d_pcd.points = o3d.utility.Vector3dVector(xyz)
            ros_cloud = convertCloudFromOpen3dToRos(open3d_pcd,'odom')
            self.inst_pub[i].publish(ros_cloud)






if __name__ == '__main__':
    scalar = 0.9                # How much gaussian spread we want 0.1 m or 10 cm
    theta_spread = PI/18.0      # 45-degrees only
    heading_bin = 12            # number of bin headings
    map_spread = 4.0
    resolution = 0.1            # XY-resolution
    #testing with random sample
    points = np.random.rand(100,3) * 10
    weights = np.random.rand(100,1)
    theta = np.random.rand(100,1) * 2*PI
    cost_map_3d_dataset = CostMapDataset(scalar=scalar, theta_spread=theta_spread,
                                         heading_bin=heading_bin, map_spread= map_spread, resolution=resolution)
    heading_weights = cost_map_3d_dataset.calculateHeading(points, weights, theta)
    print(heading_weights.shape)
    print(heading_weights[:5,:].T)

