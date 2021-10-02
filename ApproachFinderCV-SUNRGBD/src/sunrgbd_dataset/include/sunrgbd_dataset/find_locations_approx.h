//
// Created by vcr on 2020-12-10.
//

#ifndef SIMULATION_WS_FIND_LOCATIONS_APPROX_H
#define SIMULATION_WS_FIND_LOCATIONS_APPROX_H


#include "iostream"
#include "dataset_utils.h"



#define PADDING_OFFSET 1.0  // 80cm away form table
#define WHEELCHAIR_SAMPLING_DISTANCE 0.05  // sampling at every 5 cm
#define WHEELCHAIR_WIDTH 0.7
#define WHEELCHAIR_LENGTH 0.7
#define WHEELCHAIR_DEPTH 0.8
#define PI 3.14159265
#define CLUSTER_TOLERANCE WHEELCHAIR_WIDTH/2.0 //25 cm - this is wheelchair length/2
#define MIN_CLUSTER_SIZE 1
#define DECREASE_RANGE 0.124  // 1/4th of wheelchair width


//#define VISUALISATIONS
#define EUCLIDEAN_DISTANCE_THRESHOLD 0.05  // 10cms
#define EUCLIDEAN_CLUSTER_MIN_SIZE 120

//--------------------------------------------------------------------------------
#define OBJECTNESS_THRESHOLD 0.7          //tables above this threshold are only considered for plane fitting
#define NUM_OF_POINTS_FOR_OVERLAPP 10               // number of points required to considered an overlap of two BB
#define PLANE_DISTANCE_THRESHOLD 0.02       // Points above and below 2 cms are considered as table top
#define STATISTICAL_OUTLIER_NEIGHBOURS 50    // The number of neighbors to analyze for each point is set to 50
#define STATISTICAL_OUTLIER_STD 1            // Standard Deviation multiplier is 2 - all points who have distance larger than 2 std of the mean distance to the query point is marked as outlier
#define REGION_GROWING_NEIGHBOURS 30
#define REGION_GROWING_SMOOTHNESS 3.0 / 180.0 * M_PI  // angle in radians that will be used as the allowable range for the normals deviation
#define REGION_GROWING_CURVATURE 1.0                    //curvature threshold
#define REGION_GROWING_OVERLAPPING 30

#define OVERLAP_POINTS 50 // the points box and cluster should overlap to be considered.

//--------------------------------------------------------------------------------

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

#ifdef VISUALISATIONS
//visualisations
pcl::visualization::PCLVisualizer::Ptr viewer_approx(new pcl::visualization::PCLVisualizer("3D Viewer"));
#endif

ros::Publisher pub_approx,pub_table_approx;

//simulation
std::string src_frame = "odom";
std::string des_frame = "camera_depth_optical_frame";

boost::mutex access_guard_;


class FindParkingSpots{
public:
    FindParkingSpots(ros::NodeHandle nh); //constructor
    bool findSpotsCall(sunrgbd_dataset::sunrgbd_data_srv::Request &req, sunrgbd_dataset::sunrgbd_data_srv::Response &res );
    std::vector<SalientLocation> runCVPipeline(sunrgbd_data *data);
    void fitPlanarModel(pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud, pcl::ModelCoefficients::Ptr coefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr findObjectTop(pcl::ModelCoefficients::Ptr plane_coefficients, pcl::PointCloud<pcl::PointXYZ>::Ptr box_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud);
    std::vector <Eigen::Vector3f> findMinimumAreaShape(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double height);
    std::vector <std::vector<Eigen::Vector3f>> findPossiblePlacements(std::vector <Eigen::Vector3f> &approx_polygon, double padding_offset, double location_sampling_distance, float height);
    std::vector <Eigen::Vector3f> calculatePaddedLines(std::vector <Eigen::Vector3f> &approx_polygon, double padding_offset, float height);
    std::vector <std::vector<Eigen::Vector3f>> filterForFOV(std::vector <std::vector<Eigen::Vector3f>> &locations, sensor_msgs::CameraInfo cam_info);
    std::vector <std::vector<Eigen::Vector3f>> filterForCollision(std::vector <std::vector<Eigen::Vector3f>> &locations, std::vector<double> &wheelchair_dimensions, PointCloud::Ptr scene_cloud);
    std::vector<std::vector<Eigen::Quaternionf>> calculateHeading(std::vector<std::vector<Eigen::Vector3f>> &locations);
    std::vector <std::vector<double>> findPositionalWeights(std::vector <std::vector<Eigen::Vector3f>> &desirable_locations);
    std::vector <std::vector<double>> calculateVisibilityWeights(std::vector <std::vector<Eigen::Vector3f>> locations, std::vector<std::vector<Eigen::Quaternionf>> &heading,
                                                                 std::vector<double> &wheelchair_dimensions, cv::Mat depth_image, cv::Mat image, sensor_msgs::CameraInfo cam_info);
    PointCloud::Ptr cropBounds(PointCloud::Ptr cloud, double z_min, double z_max);
    std::vector<Eigen::Vector3f> addInbetweenPaddedLines(std::vector<Eigen::Vector3f> padded_approx_polygon);

protected:
    ros::ServiceServer sunrgbd_service_;
    int marker_id;
};




#endif //SIMULATION_WS_FIND_LOCATIONS_APPROX_H
