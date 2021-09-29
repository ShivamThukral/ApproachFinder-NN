//
// Created by vcr on 2021-08-14.
//

#ifndef SIMULATION_WS_DATASET_UTILS_H
#define SIMULATION_WS_DATASET_UTILS_H
#include "ros/ros.h"
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <eigen_conversions/eigen_msg.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include "dataset_generation/sunrgbd_data_srv.h"
//#include "desirable_locations/detectionArray.h"
//#include "desirable_locations/votenetDetection.h"
//#include "autorally_msgs/desiredLocation.h"
//#include "desirable_locations/locationArray.h"
#include <geometry_msgs/Point.h>
#include "algorithm"
#include <math.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl/recognition/linemod/line_rgbd.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <limits>
#include <geometry_msgs/Transform.h>
#include <boost/thread/mutex.hpp>
#include <pcl/filters/passthrough.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;



class SalientLocation{
public:
    geometry_msgs::Point location;
    double weight;
    geometry_msgs::Quaternion heading;
};


void visualiseDepthImage(cv::Mat depth_image)
{
    cv::Mat normalized;
    double max = 0.0;
    cv::minMaxLoc(depth_image, 0, &max, 0, 0);
    depth_image.convertTo(normalized, CV_32F, 1.0 / max, 0);
    cv::imshow("depth_image", normalized);
    cv::waitKey(0);
}

void visualiseImage(cv::Mat image)
{
    cv::imshow("image", image);
    cv::waitKey(0);
}

void simpleVis(PointCloud::Ptr cloud, std::string title) {
    // --------------------------------------------
    // -----Open 3D viewer xyz and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer_approx(new pcl::visualization::PCLVisualizer(title));
    viewer_approx->setBackgroundColor(0, 0, 0);
    viewer_approx->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer_approx->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer_approx->addCoordinateSystem(1.0);
    viewer_approx->initCameraParameters();
    while (!viewer_approx->wasStopped()) {
        viewer_approx->spinOnce(100);
        sleep(0.1);
    }
}



class sunrgbd_data{
public:
    float floor_height;
    PointCloud::Ptr scene_cloud;
    PointCloud::Ptr object_cloud;
    cv::Mat image;
    cv::Mat depth_image;
    sensor_msgs::CameraInfo cam_info;
    sunrgbd_data(dataset_generation::sunrgbd_data_srv::Request req);

    void visualise();
    void printDepthValues();
};

sunrgbd_data::sunrgbd_data(dataset_generation::sunrgbd_data_srv::Request req) {
    scene_cloud = PointCloud::Ptr(new PointCloud);
    sensor_msgs::PointCloud2 cloud = req.scene_pc;
    pcl::fromROSMsg(cloud, *scene_cloud);  // assign the scene cloud

    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(req.image, "8UC3");
    image = cv_ptr->image.clone();

    cv_ptr = cv_bridge::toCvCopy(req.depth_image,"32FC1");
    cv_ptr->image.convertTo(depth_image, CV_32F);

    sensor_msgs::PointCloud2 obj = req.object_pc;
    object_cloud = PointCloud::Ptr(new PointCloud);
    pcl::fromROSMsg(obj, *object_cloud);  // assgin the scene cloud

    floor_height = req.floor_height;

    cam_info = req.calib_matrix;
}

void sunrgbd_data::visualise() {
    //scene cloud
    simpleVis(scene_cloud,"scene_pc");
    //rgb image
    visualiseImage(image);
    visualiseDepthImage(depth_image);
    //object point cloud data
    simpleVis(object_cloud, "object_pc");
}

void sunrgbd_data::printDepthValues() {
    for(int i=0;i<depth_image.rows;i++)
    {
        for(int j=0;j<depth_image.cols;j++)
            cout<<depth_image.at<float>(i,j)<<" , ";
        cout<<endl;
    }
}

cv::Point3d projectPoint(cv::Point3d p, sensor_msgs::CameraInfo cam_info)
{
    cv::Matx33f rotation_matrix = cv::Matx33f(cam_info.R[0], cam_info.R[1], cam_info.R[2],
                                              cam_info.R[3], cam_info.R[4], cam_info.R[5],
                                              cam_info.R[6], cam_info.R[7], cam_info.R[8]);

    //transpose the matrix
    rotation_matrix = rotation_matrix.t();

    cv::Matx33f intrinsic_matrix = cv::Matx33f(cam_info.K[0], cam_info.K[1], cam_info.K[2],
                                               cam_info.K[3], cam_info.K[4], cam_info.K[5],
                                               cam_info.K[6], cam_info.K[7], cam_info.K[8]);
    cv::Matx13f pt_3d(p.x, p.y, p.z);
    float v1 = rotation_matrix.row(0).dot(pt_3d);
    float v2 = rotation_matrix.row(1).dot(pt_3d);
    float v3 = rotation_matrix.row(2).dot(pt_3d);
    cv::Matx13f pt_r(v1,v2,v3);
    cv::Matx13f pt_r_flipped(v1,-v3,v2);
    float u = pt_r_flipped.dot(intrinsic_matrix.row(0));
    float v = intrinsic_matrix.row(1).dot(pt_r_flipped);
    float z = intrinsic_matrix.row(2).dot(pt_r_flipped);
    u /= z;
    v /= z;

    /* Lastly flip the axis to camera --->Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward */
    cv::Point3d cam_point(p.x, -p.z, p.y);
    cv::Point3d uv(u,v,cam_point.z);
    return uv;

}

void convertPCDToDepth(PointCloud::Ptr scene_cloud, cv::Mat image, cv::Mat depth_image, sensor_msgs::CameraInfo cam_info)
{
    cv::Matx33f rotation_matrix = cv::Matx33f(cam_info.R[0], cam_info.R[1], cam_info.R[2],
                                              cam_info.R[3], cam_info.R[4], cam_info.R[5],
                                              cam_info.R[6], cam_info.R[7], cam_info.R[8]);

    //transpose the matrix
    rotation_matrix = rotation_matrix.t();

    cv::Matx33f intrinsic_matrix = cv::Matx33f(cam_info.K[0], cam_info.K[1], cam_info.K[2],
                                               cam_info.K[3], cam_info.K[4], cam_info.K[5],
                                               cam_info.K[6], cam_info.K[7], cam_info.K[8]);

    cv::Mat depth = cv::Mat::zeros(image.rows, image.cols, depth_image.type());
    for(int i = 0;i<scene_cloud->points.size();i++)
    {
        pcl::PointXYZ p=scene_cloud->points.at(i);
        cv::Matx13f pt_3d(p.x, p.y, p.z);
        float v1 = rotation_matrix.row(0).dot(pt_3d);
        float v2 = rotation_matrix.row(1).dot(pt_3d);
        float v3 = rotation_matrix.row(2).dot(pt_3d);
        cv::Matx13f pt_r(v1,v2,v3);
        cv::Matx13f pt_r_flipped(v1,-v3,v2);
        float u = pt_r_flipped.dot(intrinsic_matrix.row(0));
        float v = intrinsic_matrix.row(1).dot(pt_r_flipped);
        float z = intrinsic_matrix.row(2).dot(pt_r_flipped);
        u /= z;
        v /= z;
        cv::Point2d uv(u,v);
        if (uv.x >= 0 && uv.x <= cam_info.width && uv.y >= 0 && uv.y <= cam_info.height)
            cv::circle( image, uv, 1, cv::Scalar( 255, 255, 255 ), CV_FILLED);
    }
    cv::imshow("depth_image", image);
    cv::waitKey(4000);
}


////donot pass by reference here.
void projectLocationsOnDepthImage (std::vector<std::vector < Eigen::Vector3f>> locations, cv::Mat image, sensor_msgs::CameraInfo cam_info, std::string window_str)
{
    cv::Mat normalized = image.clone();
    cv::Matx33f rotation_matrix = cv::Matx33f(cam_info.R[0], cam_info.R[1], cam_info.R[2],
                                              cam_info.R[3], cam_info.R[4], cam_info.R[5],
                                              cam_info.R[6], cam_info.R[7], cam_info.R[8]);

    //transpose the matrix
    rotation_matrix = rotation_matrix.t();

    cv::Matx33f intrinsic_matrix = cv::Matx33f(cam_info.K[0], cam_info.K[1], cam_info.K[2],
                                               cam_info.K[3], cam_info.K[4], cam_info.K[5],
                                               cam_info.K[6], cam_info.K[7], cam_info.K[8]);

    for(int i = 0; i<locations.size(); i++)
    {
        for( int j = 0; j<locations[i].size(); j++)
        {
            cv::Matx13f pt_3d(locations[i][j][0], locations[i][j][1], locations[i][j][2]);
            float v1 = rotation_matrix.row(0).dot(pt_3d);
            float v2 = rotation_matrix.row(1).dot(pt_3d);
            float v3 = rotation_matrix.row(2).dot(pt_3d);
            cv::Matx13f pt_r(v1,v2,v3);
            cv::Matx13f pt_r_flipped(v1,-v3,v2);
            float u = pt_r_flipped.dot(intrinsic_matrix.row(0));
            float v = intrinsic_matrix.row(1).dot(pt_r_flipped);
            float z = intrinsic_matrix.row(2).dot(pt_r_flipped);
            u /= z;
            v /= z;
            cv::Point2d uv(u,v);
            if (uv.x >= 0 && uv.x <= cam_info.width && uv.y >= 0 && uv.y <= cam_info.height)
                cv::circle( normalized, uv, 1, cv::Scalar( 255, 255, 255 ), CV_FILLED);
        }
    }
    cv::namedWindow(window_str, CV_WINDOW_AUTOSIZE);
    cv::imshow(window_str, normalized);
    cv::waitKey(400);
}




#endif //SIMULATION_WS_DATASET_UTILS_H
