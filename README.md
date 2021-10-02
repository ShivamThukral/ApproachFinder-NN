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
Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

To see if the compilation is successful, try to run `python models/backbone_module.py`, `python models/voting_module.py`, `python models/my_proposal_module.py` and  `python models/parknet.py` to see if a forward pass works.

## TODO:Demo
We ship the code with a pretrained model under *'sunrgbd_demo/model/checkpoint.tar'*. We also ship some sample point clouds for running this demo (refer *sunrgdb_demo/demo_files*).
Run the following command for demo:
```python
python demo.py
```
For more information about input use `-h` option. The demo uses a pre-trained model to detect potential docking locations from validation set of SUN RGB-D dataset. We visualise 3D results using open3D and dump relevant intermendiate results in the dump folder.

## Training
**Dataset Preparation:** follow the instructions [here](ApproachFinderCV-SUNRGBD/README.md)

To train ApproachFinder-NN on SUN RGB-D data:
```python
python train.py --log_dir log_docknet
```
Use `-h` option to know more about training options. While training you can use TenserBoard to see loss curves by running `python -m tensorboard.main --logdir=<log_dir_name> --port=6006`
## Evaluation
We evaluate our network on two criteria: 3D-bounding box tightness and desirability costmap. For further details about these mentrics please refer the paper.
```python
python eval.py --checkpoint_path log_docknet/checkpoint.tar --dump_dir eval_docknet
python eval_desirability.py --checkpoint_path log_docknet/checkpoint.tar --dump_dir eval_des_docknet
```
Example results will be dumped in the eval_docknet and eval_des_docknet folder. Please use `-h` option to learn more about evaluation options. 
The evaluation results are stored in log_eval.txt file for both the metrics. 

## Results on SUN RGB-D dataset
