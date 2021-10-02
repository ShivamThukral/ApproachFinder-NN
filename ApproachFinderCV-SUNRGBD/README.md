# Dataset Preparation
### Prepare SUN RGB-D Data

**Download and extract the data**
1. Download SUNRGBD v2 data from [here](http://rgbd.cs.princeton.edu/data/) (SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat) and the toolkits (SUNRGBDtoolbox.zip). Move all the downloaded files under *"dockent/sunrgbd/OFFICIAL_SUNRGBD"*. Extract the files here.

2. Extract point clouds and annotations (class, v2 2D -- xmin,ymin,xmax,ymax, and 3D bounding boxes -- centroids, size, 2D heading) by running `extract_split.m` and `extract_rgbd_data_v2.m` under the `matlab` folder.

3. Prepare data by running `python sunrgbd_data.py --gen_v2_data`

4. Visualise the results by running `python my_sunrgbd_data.py --viz` and  use MeshLab to view the generated PLY files at `data_viz_dump`. 

NOTE: SUNRGBDtoolbox.zip should have MD5 hash `18d22e1761d36352f37232cba102f91f` (you can check the hash with `md5 SUNRGBDtoolbox.zip` on Mac OS or `md5sum SUNRGBDtoolbox.zip` on Linux)


### Run ApproachFinder-CV on SUN RGB-D dataset.
1. Compile the ROS-Project
```asm
cd ApproachFinderCV-SUNRGBD
catkin_make
source ./devel/setup.bash
```
2. Run the following commands in separate terminals:
```asm
rosrun sunrgbd_generation find_parking_spots_dataset
rosrun sunrgbd_generation sunrgbd_dataset_builder.py
```

This will create two folders in the sunrgbd_generation package: