//
// Created by vcr on 2021-01-08.
//
#include "sunrgbd_dataset/find_locations_approx.h"

std::vector <std::vector<Eigen::Vector3f>>
FindParkingSpots::filterForFOV(std::vector <std::vector<Eigen::Vector3f>> &locations, sensor_msgs::CameraInfo cam_info) {
    std::vector <std::vector<Eigen::Vector3f>> locations_FOV_filtered;
    cv::Matx33f rotation_matrix = cv::Matx33f(cam_info.R[0], cam_info.R[1], cam_info.R[2],
                                              cam_info.R[3], cam_info.R[4], cam_info.R[5],
                                              cam_info.R[6], cam_info.R[7], cam_info.R[8]);

    //transpose the matrix
    rotation_matrix = rotation_matrix.t();

    cv::Matx33f intrinsic_matrix = cv::Matx33f(cam_info.K[0], cam_info.K[1], cam_info.K[2],
                                               cam_info.K[3], cam_info.K[4], cam_info.K[5],
                                               cam_info.K[6], cam_info.K[7], cam_info.K[8]);
    int total = 0;
    for(int i = 0; i<locations.size(); i++)
    {
        std::vector <Eigen::Vector3f> filter_FOV;
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
                filter_FOV.push_back(locations[i][j]);
        }
        //add edge only if we have some valid locations
        if(filter_FOV.size() >= 2)   //atleast tow since we need tow points in the next step
            locations_FOV_filtered.push_back(filter_FOV);
        total += filter_FOV.size();
    }
    cout << "FOV filtered locations are : " << total << endl;
    return locations_FOV_filtered;
}

std::vector <Eigen::Vector3f>
FindParkingSpots::calculatePaddedLines(std::vector <Eigen::Vector3f> &approx_polygon, double padding_offset, float height) {
    std::vector <Eigen::Vector3f> padded_edges;
    std::vector <cv::Point2f> contour;
    for (Eigen::Vector3f point:approx_polygon)
        contour.push_back(cv::Point2f(point[0], point[1]));
    contour.pop_back();

    for (int i = 0; i < approx_polygon.size() - 1; i++) {
        cv::Point2f p1 = cv::Point2f(approx_polygon[i][0],
                                     approx_polygon[i][1]); // "start"  // z point is not required for our case
        cv::Point2f p2 = cv::Point2f(approx_polygon[i + 1][0], approx_polygon[i + 1][1]); // "end"

        // take care with division by zero caused by vertical lines
        double slope = (p2.y - p1.y) / (double) (p2.x - p1.x);
        double perpendicular_slope = -1.0 / (slope);
        cv::Point2f padded_point1, padded_point2, padded_point3, padded_point4;
        cv::Point2f padded_point1_smaller, padded_point2_smaller, padded_point3_smaller, padded_point4_smaller;

        padded_point1.x = p1.x + sqrt(pow(padding_offset, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point1.y = perpendicular_slope * (padded_point1.x - p1.x) + p1.y;
        padded_point2.x = p1.x - sqrt(pow(padding_offset, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point2.y = perpendicular_slope * (padded_point2.x - p1.x) + p1.y;
        padded_point3.x = p2.x + sqrt(pow(padding_offset, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point3.y = perpendicular_slope * (padded_point3.x - p2.x) + p2.y;
        padded_point4.x = p2.x - sqrt(pow(padding_offset, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point4.y = perpendicular_slope * (padded_point4.x - p2.x) + p2.y;

        double padding_offset_smaller = 0.02; // 2cm
        padded_point1_smaller.x = p1.x + sqrt(pow(padding_offset_smaller, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point1_smaller.y = perpendicular_slope * (padded_point1_smaller.x - p1.x) + p1.y;
        padded_point2_smaller.x = p1.x - sqrt(pow(padding_offset_smaller, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point2_smaller.y = perpendicular_slope * (padded_point2_smaller.x - p1.x) + p1.y;
        padded_point3_smaller.x = p2.x + sqrt(pow(padding_offset_smaller, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point3_smaller.y = perpendicular_slope * (padded_point3_smaller.x - p2.x) + p2.y;
        padded_point4_smaller.x = p2.x - sqrt(pow(padding_offset_smaller, 2.0) / (1 + pow(pow(slope, 2.0), -1.0)));
        padded_point4_smaller.y = perpendicular_slope * (padded_point4_smaller.x - p2.x) + p2.y;

        cv::Point2f mid_point1 = (padded_point1_smaller + padded_point3_smaller) / 2.0;
        cv::Point2f mid_point2 = (padded_point2_smaller + padded_point4_smaller) / 2.0;
        //double height = std::max((double) approx_polygon[i][2],WHEELCHAIR_DEPTH/2.0);

        //the z height is max of table height or wheelchair height
        if (cv::pointPolygonTest(contour, mid_point1, false) == -1) {
            padded_edges.push_back(Eigen::Vector3f(padded_point1.x, padded_point1.y, height));
            padded_edges.push_back(Eigen::Vector3f(padded_point3.x, padded_point3.y, height));
        } else {
            padded_edges.push_back(Eigen::Vector3f(padded_point2.x, padded_point2.y, height));
            padded_edges.push_back(Eigen::Vector3f(padded_point4.x, padded_point4.y, height));
        }
    }
    cout << "Points in Padded Edges : " << padded_edges.size() << endl;
    return padded_edges;
}

std::vector<Eigen::Vector3f> FindParkingSpots::addInbetweenPaddedLines(std::vector<Eigen::Vector3f> padded_approx_polygon)
{
    std::vector<Eigen::Vector3f> padded_polygon(padded_approx_polygon.size()*2);
    for(int i=0;i<padded_approx_polygon.size();i+=2)
    {
        //cout<<padded_approx_polygon[i][0]<<"\t"<<padded_approx_polygon[i][1]<<" ---> "<<padded_approx_polygon[i+1][0]<<"\t"<<padded_approx_polygon[i+1][1]<<endl;
        padded_polygon[i*2] = padded_approx_polygon[i];
        padded_polygon[i*2 + 1] = padded_approx_polygon[i+1];
        padded_polygon[i*2 + 2] = padded_approx_polygon[i+1];
        if(i!=0)
            padded_polygon[i*2-1] = padded_approx_polygon[i];
    }

    // join the last edge
    padded_polygon[padded_polygon.size()-1] = padded_approx_polygon[0];
    return padded_polygon;
}

std::vector <std::vector<Eigen::Vector3f>>
FindParkingSpots::findPossiblePlacements(std::vector <Eigen::Vector3f> &approx_polygon,
                                          double padding_offset, double location_sampling_distance, float height) {
    int total_collsion_points = 0;
    std::vector <std::vector<Eigen::Vector3f>> collsion_points;
    // add the first point back again so that we have a complete loop
    approx_polygon.push_back(approx_polygon[0]);

    std::vector <Eigen::Vector3f> padded_gapped_approx_polygon = calculatePaddedLines(approx_polygon, padding_offset, height);
    std::vector <Eigen::Vector3f> padded_approx_polygon = addInbetweenPaddedLines(padded_gapped_approx_polygon);

    for (int i = 0; i < padded_approx_polygon.size(); i = i + 2)  // last is repeated loop point so dont include it
    {

        std::vector <Eigen::Vector3f> points_on_line;
        float ratio = location_sampling_distance / sqrt((padded_approx_polygon[i] - padded_approx_polygon[i + 1]).dot(
                padded_approx_polygon[i] - padded_approx_polygon[i + 1]));
        float proportion = ratio;
        while (proportion < 1.0) {
            Eigen::Vector3f point =
                    padded_approx_polygon[i] + (padded_approx_polygon[i + 1] - padded_approx_polygon[i]) * proportion;
            points_on_line.push_back(point);
            proportion += ratio;
        }
        total_collsion_points += points_on_line.size();
        collsion_points.push_back(points_on_line);
    }

    std::cout << "Found " << total_collsion_points << " collision points along "
              << collsion_points.size() << " edges" << endl;
    return collsion_points;
}

std::vector <Eigen::Vector3f>
FindParkingSpots::findMinimumAreaShape(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double height) {
    // project the 3D point onto a 2D plane but removing the height component - assumption evey point in plane has roughly sample height
    std::vector <cv::Point2f> points;
    for (unsigned int ii = 0; ii < cloud->points.size(); ii++) {
        cv::Point2f p2d(cloud->points[ii].x, cloud->points[ii].y);
        points.push_back(p2d);
    }

    // test for circularity of the table top
    // convex hull of the table top
    std::vector <cv::Point2f> hull_points;
    std::vector <cv::Point2f> contour;  // Convex hull contour points
    cv::convexHull(points, hull_points,
                   false);  //https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656
    // Approximating polygonal curve to convex hull
    cv::approxPolyDP(hull_points, contour, 0.1,
                     true);   //https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

    std::vector <Eigen::Vector3f> locations;
    // convert this to eigen vector
    for (cv::Point2f point:contour) {
        Eigen::Vector3f p(point.x, point.y, height);
        locations.push_back(p);
    }
    std::cout << "Convex Polygon Vertices : " << locations.size() << std::endl;
    return locations;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr FindParkingSpots::findObjectTop(pcl::ModelCoefficients::Ptr plane_coefficients,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr box_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cluster(new pcl::PointCloud <pcl::PointXYZ>); //returned object

    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_pcd(new pcl::PointCloud <pcl::PointXYZ>);
    for (int i = 0; i < scene_cloud->size(); i++) {
        pcl::PointXYZ point = scene_cloud->at(i);
        float plane_value = plane_coefficients->values[0] * point.x + plane_coefficients->values[1] * point.y +
                            plane_coefficients->values[2] * point.z + plane_coefficients->values[3];
        if (abs(plane_value) <= PLANE_DISTANCE_THRESHOLD) {
            plane_pcd->push_back(point);
        }
    }
    plane_pcd->width = plane_pcd->size();
    plane_pcd->height = 1;
    plane_pcd->is_dense = true;
    //copy the points to pcd
    //simpleVis(plane_pcd,"plane_pcd");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud <pcl::PointXYZ>);
    // Apply statistical outlier for removing noise and smaller points from plane pcd
    // link https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html
    pcl::StatisticalOutlierRemoval <pcl::PointXYZ> sor;
    sor.setInputCloud(plane_pcd);
    sor.setMeanK(STATISTICAL_OUTLIER_NEIGHBOURS);
    sor.setStddevMulThresh(STATISTICAL_OUTLIER_STD);
    sor.filter(*cloud_filtered);
    //simpleVis(cloud_filtered,"statistical");

    // Region Growing segmentation
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree <pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation <pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud_filtered);
    normal_estimator.setKSearch(STATISTICAL_OUTLIER_NEIGHBOURS);
    normal_estimator.compute(*normals);

    //Region growing segmentation to find clusters in the plane pcd.
    pcl::RegionGrowing <pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);
    reg.setMaxClusterSize(25000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(REGION_GROWING_NEIGHBOURS);
    reg.setInputCloud(cloud_filtered);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(REGION_GROWING_SMOOTHNESS);
    reg.setCurvatureThreshold(REGION_GROWING_CURVATURE);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
//    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
//    pcl::visualization::CloudViewer viewer ("Cluster viewer");
//    viewer.showCloud(colored_cloud);
//    while (!viewer.wasStopped ())
//    {
//    }

    // for each cluster check which cluster has overlap with original box and merge it into returned result.
    // general approach - create a convex hull and check if points lie inside the hull or not
    pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts(new pcl::PointCloud <pcl::PointXYZ>);
    pcl::ConvexHull <pcl::PointXYZ> chull;
    std::vector <pcl::Vertices> hullPolygons;
    //create the convex hull
    chull.setInputCloud(box_pcd);
    chull.setComputeAreaVolume(true);   // compute the area and volume of the convex hull
    chull.setDimension(3);          // returns 2d convex hull - set it to 3 for XYZ plane
    chull.reconstruct(*convex_hull_pts, hullPolygons);

    pcl::CropHull <pcl::PointXYZ> cropHullFilter;
    //check within convex hull using filter
    cropHullFilter.setHullIndices(hullPolygons);
    cropHullFilter.setHullCloud(convex_hull_pts);
    cropHullFilter.setDim(3); // if you uncomment this, it will work
    //cropHullFilter.setCropOutside(false); // this will remove points inside the hull

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud <pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->push_back((*cloud_filtered)[*pit]);
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        //cloud_cluster->is_dense = true;
        //simpleVis(cloud_cluster,"cluster"+std::to_string(j));
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud <pcl::PointXYZ>);
        cropHullFilter.setInputCloud(cloud_cluster);   // taken from class which is the scene cloud
        cropHullFilter.filter(*filtered);
        if (filtered->size() >= REGION_GROWING_OVERLAPPING) {
            std::cout << filtered->size() << " Selected Cluster " << j << std::endl;
            *merged_cluster += *cloud_cluster;
        }
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
        j++;
    }
    //simpleVis(merged_cluster,"test");
    return merged_cluster;
}

void FindParkingSpots::fitPlanarModel(pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud,
                                  pcl::ModelCoefficients::Ptr coefficients) {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation <pcl::PointXYZ> seg;    // Create the segmentation object
    seg.setOptimizeCoefficients(true);     // Optional
    seg.setModelType(pcl::SACMODEL_PLANE);  // Fitting a plane on this point cloud
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(PLANE_DISTANCE_THRESHOLD);     // Minimum distance to be considered
    seg.setInputCloud(object_cloud);
    seg.segment(*inliers, *coefficients);
    ASSERT(inliers->indices.size() > 0, "Could not estimate a planar model for the given dataset.");
    // equation of the plane
    std::cout << "Model coefficients (a,b,c,d): " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;
    std::cout << "Number of inliers " << inliers->indices.size() << " points." << std::endl;
}

std::vector <std::vector<Eigen::Vector3f>>
FindParkingSpots::filterForCollision(std::vector <std::vector<Eigen::Vector3f>> &locations,
                                      std::vector<double> &wheelchair_dimensions, PointCloud::Ptr scene_cloud) {
       std::vector <std::vector<Eigen::Vector3f>> desirable_locations;
       double wheelchair_width = wheelchair_dimensions[0]-0.1, wheelchair_length = wheelchair_dimensions[1] -0.1 , wheelchair_depth = wheelchair_dimensions[2] - 0.1;

       int total_points = 0;
       for (int ii = 0; ii < locations.size(); ii++) {

           double wheelchair_rotation_radian = 0;
           Eigen::Quaternionf wheelchair_rotation;

           //the wheelchair rotation angle along each edge
           if (locations[ii].size() >= 2) {
               Eigen::Vector3f point1 = locations[ii][0], point2 = locations[ii][locations[ii].size() -
                                                                                 1]; // first and last point
               double slope = (point2[1] - point1[1]) / (point2[0] - point1[0]);
               wheelchair_rotation_radian = atan2((point2[1] - point1[1]), (point2[0] - point1[0]));
           }
           wheelchair_rotation = Eigen::AngleAxisf(wheelchair_rotation_radian,
                                                   Eigen::Vector3f::UnitZ());  // rotation along z-axis only

           // std::cout << wheelchair_rotation_radian * 180 / M_PI << std::endl;
           std::vector <Eigen::Vector3f> filtered_points;
           for (int j = 0; j < locations[ii].size(); j++) {

               //https://vtk.org/doc/nightly/html/classvtkDataSet.html
               //https://vtk.org/doc/nightly/html/classvtkPolyData.html
               //https://github.com/PointCloudLibrary/pcl/blob/master/visualization/src/common/shapes.cpp
               vtkSmartPointer <vtkDataSet> data = pcl::visualization::createCube(locations[ii][j],
                                                                                  wheelchair_rotation,
                                                                                  wheelchair_width, wheelchair_length,
                                                                                  wheelchair_depth);

               std::set <std::vector<double>> cube_corners;
               for (int i = 0;
                    i < data->GetNumberOfPoints(); i++)            // returns all the edges 12*2 = 24 bidirectional

               {
                   std::vector<double> edges{data->GetPoint(i)[0], data->GetPoint(i)[1], data->GetPoint(i)[2]};
                   cube_corners.insert(edges);
               }
               pcl::PointCloud<pcl::PointXYZ>::Ptr cube_pcd(new pcl::PointCloud <pcl::PointXYZ>);
               cube_pcd->width = 4; //8 corners
               cube_pcd->height = 2;
               for (std::vector<double> corner:cube_corners) {
                   pcl::PointXYZ point(corner[0], corner[1], corner[2]);
                   cube_pcd->push_back(point);
               }

               // general approach - create a convex hull and check if points lie inside the hull or not
               pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts(new pcl::PointCloud <pcl::PointXYZ>);
               pcl::ConvexHull <pcl::PointXYZ> chull;
               std::vector <pcl::Vertices> hullPolygons;
               //create the convex hull
               chull.setInputCloud(cube_pcd);
               chull.setComputeAreaVolume(true);   // compute the area and volume of the convex hull
               chull.setDimension(3);          // returns 2d convex hull - set it to 3 for XYZ plane
               chull.reconstruct(*convex_hull_pts, hullPolygons);

               pcl::CropHull <pcl::PointXYZ> cropHullFilter;
               //check within convex hull using filter
               cropHullFilter.setHullIndices(hullPolygons);
               cropHullFilter.setHullCloud(convex_hull_pts);
               cropHullFilter.setDim(3); // if you uncomment this, it will work
               //cropHullFilter.setCropOutside(false); // this will remove points inside the hull

               //optimised-  check for each point invidually
               pcl::PointCloud<pcl::PointXYZ>::Ptr check_cloud(new pcl::PointCloud <pcl::PointXYZ>);
               // Fill in the cloud data
               check_cloud->width = 1;
               check_cloud->height = 1;
               check_cloud->points.resize(check_cloud->width * check_cloud->height);
               bool point_inside = false;
               for (int nIndex = 0; nIndex < scene_cloud->points.size() && !point_inside; nIndex++) {
                   pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud <pcl::PointXYZ>);
                   check_cloud->points[0] = scene_cloud->points[nIndex];
                   cropHullFilter.setInputCloud(check_cloud);   // taken from class which is the scene cloud
                   cropHullFilter.filter(*filtered);
                   if (filtered->size() > 0) {
                       point_inside = true;
                   }
               }
               if (!point_inside) {
                   filtered_points.push_back(locations[ii][j]);      // if no point inside the convex hull
               }

           }
           if(filtered_points.size() >= 2 )
           {
               total_points += filtered_points.size();
               desirable_locations.push_back(filtered_points);
           }

       }
       std::cout << "Found " << total_points << " non colliding  points along " << desirable_locations.size()
                 << " edges" << endl;
       return desirable_locations;

}

std::vector<std::vector<Eigen::Quaternionf>> FindParkingSpots::calculateHeading(std::vector<std::vector<Eigen::Vector3f>> &locations){
    std::vector<std::vector<Eigen::Quaternionf>> heading;
    for(int i=0;i<locations.size();i++)
    {
        Eigen::Quaternionf wheelchair_rotation;
        double wheelchair_rotation_radian = -0;
        //the wheelchair rotation angle along each edge
        if(locations[i].size() >= 2){
            Eigen::Vector3f point1 = locations[i].front(), point2 = locations[i].back(); // first and last point
            double slope = (point2[1]-point1[1])/(point2[0]-point1[0]);
            wheelchair_rotation_radian = atan2((point2[1]-point1[1]),(point2[0]-point1[0]));
            wheelchair_rotation_radian += -1.57;  // to make the array perpendicular
        }
        wheelchair_rotation = Eigen::AngleAxisf(wheelchair_rotation_radian,
                                                Eigen::Vector3f::UnitZ());  // rotation along z-axis only

        std::vector<Eigen::Quaternionf> edge_heading(locations[i].size(), wheelchair_rotation);
        heading.push_back(edge_heading);
    }
    return heading;
}

std::vector <std::vector<double>>
FindParkingSpots::findPositionalWeights(std::vector <std::vector<Eigen::Vector3f>> &desirable_locations) {
    /*
    * Points which are close to corners and other chairs should be weighted less
    */
    std::vector <std::vector<double>> weights; // final value to be returned
    for(std::vector<Eigen::Vector3f> &locations:desirable_locations)
    {
        std::vector<double> weight_in_cluster(locations.size(), 1);
        weights.push_back(weight_in_cluster);
    }
    return weights;
}

std::vector <std::vector<double>>
FindParkingSpots::calculateVisibilityWeights(std::vector <std::vector<Eigen::Vector3f>> locations, std::vector<std::vector<Eigen::Quaternionf>> &heading,
                                              std::vector<double> &wheelchair_dimensions, cv::Mat depth_image, cv::Mat image, sensor_msgs::CameraInfo cam_info) {
    //returned value
    std::vector <std::vector<double>> visibility_weights;
    cv::Mat normalized;
    depth_image.copyTo(normalized);

    double wheelchair_width = wheelchair_dimensions[0] , wheelchair_length = wheelchair_dimensions[1] , wheelchair_depth = wheelchair_dimensions[2];

    //for each location
    for (int ii = 0; ii < locations.size(); ii++) {

        std::vector<double> cluster_weight;
        cout << locations[ii].size() << " : ";
        for (int j = 0; j < locations[ii].size(); j++) {
            double current_weight = 0.0;
            cv::Point3d pt_3d(locations[ii][j][0], locations[ii][j][1], locations[ii][j][2]);
            cv::Point3d uv = projectPoint(pt_3d, cam_info);

            //center is always visible
            if (uv.x >= 0 && uv.x <= cam_info.width && uv.y >= 0 && uv.y <= cam_info.height) {

                //cv::circle( normalized, cv::Point2d(uv.x,uv.y), 5, 8, CV_FILLED);
                float raw = depth_image.at<float>(uv.y, uv.x);
                //cout<<uv<< " \t" << raw << " , " << locations[ii][j][2] << endl;
                // if the point is above the depth point then its visible
                if (std::isnan(raw) || raw >= uv.z) {
                    current_weight += 0.2;
                    //cv::circle( normalized, uv, 2, cv::Scalar( 0, 255, 255 ), CV_FILLED);
                }
            }

            vtkSmartPointer <vtkDataSet> data = pcl::visualization::createCube(locations[ii][j], heading[ii][j],
                                                                               wheelchair_width, wheelchair_length,
                                                                               wheelchair_depth);
            std::set <std::vector<double>> cube_corners;
            for (int i = 0;i < data->GetNumberOfPoints(); i++) {            // returns all the edges 12*2 = 24 bidirectional
                std::vector<double> edges{data->GetPoint(i)[0], data->GetPoint(i)[1], data->GetPoint(i)[2]};
                cube_corners.insert(edges);
            }

            for (std::set < std::vector < double >> ::iterator it = cube_corners.begin(); it != cube_corners.end(); it++)
            {
                cv::Point3d pt;
                pt.x = (*it)[0], pt.y = (*it)[1], pt.z = (*it)[2];
                cv::Point3d uv = projectPoint(pt, cam_info);

                if (uv.x >= 0 && uv.x <= cam_info.width && uv.y >= 0 && uv.y <= cam_info.height) {
                    //cv::circle( normalized, cv::Point2d(uv.x,uv.y), 5, 8, CV_FILLED);
                    // if the point is above the depth point then its visible
                    float raw = depth_image.at<float>(uv.y, uv.x);
                    if (std::isnan(raw) || raw >= uv.z)
                        current_weight += 0.1;
                }
            }
            cout<<current_weight << " , ";
            cluster_weight.push_back(current_weight);
        }
        cout<<endl;
        visibility_weights.push_back(cluster_weight);
    }
    //visualiseDepthImage(normalized);
   // projectLocationsOnDepthImage(locations, cam)
    return visibility_weights;
}

PointCloud::Ptr FindParkingSpots::cropBounds(PointCloud::Ptr cloud, double z_min, double z_max)
{
    PointCloud::Ptr cloud_filtered_xyz (new PointCloud);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (z_min, z_max);
    pass.filter (*cloud_filtered_xyz);
    return cloud_filtered_xyz;
}

std::vector<SalientLocation> FindParkingSpots::runCVPipeline(sunrgbd_data *data)
{
    //return the values to the client
    std::vector<SalientLocation> salient_locations;
#ifdef VISUALISATIONS
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_coloured(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::copyPointCloud(*data->scene_cloud, *scene_coloured);
#endif
    //Here, we don't need to do region growing segmentation as the bboxes are perfect.
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_top_pcd = data->object_cloud;
/*    // Try to fit a planar model in the plane.
    pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
    // gets the equation of the plane for object top/ table top
    fitPlanarModel(data->object_cloud, plane_coefficients);   // returns the table_top in terms of plane
    //find the table top based on region growing segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_top_pcd = findObjectTop(plane_coefficients, data->object_cloud, data->scene_cloud);
    //too small table pcds can be ignored.
    if(object_top_pcd->size() <= 10)
    {
        std::cerr<<"Plane PCD too small.. skipping this object";
        return salient_locations;
    }*/

#ifdef VISUALISATIONS
        for(int i=0;i<scene_coloured->size();i++) {
            pcl::PointXYZ point = data->scene_cloud->at(i);
            scene_coloured->points[i].r = 170;
            scene_coloured->points[i].g = 175;
            scene_coloured->points[i].b = 175;
/*          float plane_value = plane_coefficients->values[0]*point.x + plane_coefficients->values[1]*point.y + plane_coefficients->values[2]*point.z + plane_coefficients->values[3];
            if(abs(plane_value) <= PLANE_DISTANCE_THRESHOLD)
            {
                scene_coloured->points[i].g = 125;
            }*/
        }
        // paint the table top separately - THIS IS A BUG IN THE CODE
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_top_colored(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(*object_top_pcd, *object_top_colored);
        *scene_coloured += *object_top_colored;
#endif
    double height = data->floor_height + WHEELCHAIR_DEPTH/2.0  + 0.15; //additinal offset
    std::vector <Eigen::Vector3f> polygon_points = findMinimumAreaShape(object_top_pcd, height);
    std::vector <std::vector<Eigen::Vector3f>> locations = findPossiblePlacements(polygon_points, PADDING_OFFSET / 2.0, WHEELCHAIR_SAMPLING_DISTANCE, height);

#ifdef VISUALISATIONS
        for(int i=0;i<polygon_points.size();i++){
               pcl::PointXYZ sphere_center(polygon_points[i][0],polygon_points[i][1],polygon_points[i][2]);
               viewer_approx->addSphere(sphere_center,0.1,"bbox"+std::to_string(this->marker_id++));
            }
        for(int i=0;i<locations.size();i++){
            for(int j=0;j<locations[i].size();j++){
                    pcl::PointXYZ sphere_center(locations[i][j][0],locations[i][j][1],locations[i][j][2]);
                    //viewer_approx->addSphere(sphere_center,0.04,"bbox"+std::to_string(marker_id++));
                }
            }
#endif
    //convertPCDToDepth(data->scene_cloud, data->image, data->depth_image, data->cam_info);
    //projectLocationsOnDepthImage(locations, data->image,data->cam_info);
    std::vector <std::vector<Eigen::Vector3f>> locations_filtered = filterForFOV(locations, data->cam_info);
    PointCloud::Ptr ground_removed = cropBounds(data->scene_cloud , data->floor_height + 0.09, data->floor_height + 2.0); //remove the ground plane
    std::vector<double> wheelchairDimensions = {WHEELCHAIR_WIDTH, WHEELCHAIR_LENGTH, WHEELCHAIR_DEPTH};
    std::vector <std::vector<Eigen::Vector3f>> desirable_locations = filterForCollision(locations_filtered, wheelchairDimensions, ground_removed);
    std::vector<std::vector<Eigen::Quaternionf>> heading = calculateHeading(desirable_locations);
    // find weights for each desirable location
    std::vector <std::vector<double>> position_weights = findPositionalWeights(desirable_locations);
    std::vector <std::vector<double>> visibility_weights = calculateVisibilityWeights(desirable_locations, heading, wheelchairDimensions, data->depth_image, data->image, data->cam_info);


#ifdef VISUALISATIONS
    projectLocationsOnDepthImage(locations_filtered, data->image, data->cam_info, "FOV_Filtered");
    projectLocationsOnDepthImage(desirable_locations, data->image, data->cam_info, "Collision_Filtered");
    for(int i=0;i<desirable_locations.size();i++)
        {
          Eigen::Quaternionf wheelchair_rotation;
          double wheelchair_rotation_radian = -0;
          //the wheelchair rotation angle along each edge
        if(desirable_locations[i].size() >= 2){
            Eigen::Vector3f point1 = desirable_locations[i].front(), point2 = desirable_locations[i].back(); // first and last point
            double slope = (point2[1]-point1[1])/(point2[0]-point1[0]);
            wheelchair_rotation_radian = atan2((point2[1]-point1[1]),(point2[0]-point1[0]));
            wheelchair_rotation_radian += -1.57;  // to make the array perpendicular
        }
         wheelchair_rotation = Eigen::AngleAxisf(wheelchair_rotation_radian,
                                            Eigen::Vector3f::UnitZ());  // rotation along z-axis only
        for(int j=0;j<desirable_locations[i].size();j++)
            {
                pcl::PointXYZ sphere_center(desirable_locations[i][j][0],desirable_locations[i][j][1],desirable_locations[i][j][2]);
                viewer_approx->addSphere(sphere_center,0.04,"bbox"+std::to_string(marker_id++));
                pcl::PointXYZ arrow_end(sphere_center.x - 0.5*cos(wheelchair_rotation_radian),sphere_center.y - 0.5*sin(wheelchair_rotation_radian),sphere_center.z);
                viewer_approx->addArrow(arrow_end,sphere_center,1,0,0,false,"arrow"+std::to_string(marker_id++));
             //   viewer_approx->addCube(desirable_locations[i][j],wheelchair_rotation,0.04,0.04,0.04,"cube"+std::to_string(this->marker_id++));
            }
        }
    viewer_approx->setBackgroundColor(255, 255, 255);
   // viewer_approx->resetCamera();
    viewer_approx->addPointCloud<pcl::PointXYZRGB> (scene_coloured, "sample_cloud" );
    viewer_approx->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample_cloud");
    viewer_approx->addCoordinateSystem (1.0);
    viewer_approx->initCameraParameters ();
//    viewer_approx->removeCoordinateSystem();

    while (!viewer_approx->wasStopped()) {
        viewer_approx->spinOnce(100);
        sleep(0.1);
    }
    viewer_approx->removeAllPointClouds();
    viewer_approx->resetStoppedFlag();
    viewer_approx->removeAllShapes();
#endif
    //assert that size match
    for (int i = 0; i < position_weights.size(); i++) {
        ASSERT(position_weights[i].size() == visibility_weights[i].size(),
               "Mismatch Visibility and Position Weights size");
        ASSERT(desirable_locations[i].size() == heading[i].size(), "Mismatch in Location and heading size");
        for(int j = 0;j<position_weights[i].size();j++) {
            position_weights[i][j] *= visibility_weights[i][j];
        }
    }


    for (int i = 0; i < desirable_locations.size(); i++) {
        for (int j = 0; j < desirable_locations[i].size(); j++) {
            assert(desirable_locations[i].size() == position_weights[i].size());
            assert(desirable_locations[i].size() == heading[i].size());

            SalientLocation loc;
            //assign the parking spots
            loc.location.x = desirable_locations[i][j][0];
            loc.location.y = desirable_locations[i][j][1];
            loc.location.z = desirable_locations[i][j][2];
            //assign the weight
            loc.weight = position_weights[i][j];
            //heading
            tf::quaternionEigenToMsg(heading[i][j].cast<double>(), loc.heading);
            salient_locations.push_back(loc);
        }
    }

   return salient_locations;
}

FindParkingSpots::FindParkingSpots(ros::NodeHandle nh)
{
    //Initialise Server
    sunrgbd_service_ = nh.advertiseService("sunrgbd_data_groundtruth", &FindParkingSpots::findSpotsCall, this);
    marker_id = 0;
}

bool FindParkingSpots::findSpotsCall(sunrgbd_dataset::sunrgbd_data_srv::Request &req, sunrgbd_dataset::sunrgbd_data_srv::Response &res )
{
    sunrgbd_data *data = new sunrgbd_data(req);
    //visulise the data
    //data->visualise();
    //data->printDepthValues();

    //run the cv pipeline to generate the parking locations
    std::vector<SalientLocation> locations = runCVPipeline(data);

    //response to be returned
    res.success = locations.size() > 0;
    for(SalientLocation &loc:locations)
    {
        sunrgbd_dataset::desiredLocation des_loc;
        des_loc.location = loc.location;
        des_loc.location_weight = loc.weight;
        des_loc.heading = loc.heading;
        res.desired_locations.push_back(des_loc);
    }
    //clean the memory
    delete(data);
    return true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "dataset_parking_spots");
    ros::NodeHandle nh;
    FindParkingSpots *parking_spots = new FindParkingSpots(nh);
    ros::spin();
    //clean-up
    delete(parking_spots);
    return 0;
}


