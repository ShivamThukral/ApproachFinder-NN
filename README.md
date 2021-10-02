# ApproachFinder-NN
A real-time computer vision algorithm to find potential docking locations indoor environments.

## Project Brief:
TODO: Add abstract here

## Installation Instructions:
Please follow the installation instructions mentioned [here](https://github.com/ShivamThukral/ApproachFinder-CV#installation-instructions) to run the ApproachFinder-CV pipeline. Apart from this, you will have to install the following packages to run ApproachFInder-NN:
- Install [MATLAB](https://www.mathworks.com/help/install/) extract SUN RGB-D dataset. 
- Install python 3.8. It's recommended that you use [Anaconda](https://www.anaconda.com/products/individual) for your installations.
- Install [Pytorch](https://pytorch.org/)
    - `conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch`
- Install [Tensorflow](https://github.com/tensorflow/tensorflow) for [TensorBoard](https://www.tensorflow.org/tensorboard)

This code is compatible with Pytorch v1.4, CUDA 10.1 and TensorFlow v2.x 

Install the following Python dependencies (with `pip install`):
```asm
matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
Pillow
```

Install the following packages with conda:

conda install numpy

conda install -c anaconda pillow

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install -c open3d-admin -c conda-forge open3d

conda install -c conda-forge rospkg
