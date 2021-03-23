# matObjPose
## Introduction
This repository is the implementation of pose detection algorithms presented in our SII 2019 paper "Toward 6 DOF Object Pose Estimation with Minimum Dataset".
We provide MATLAB codes for training and testing the convolutional neural network to estimate 
a 3D orientation of the object from a 2D image. Also, shape registration based on iterative closest point 
is included.

## Environment 
MATLAB with Image Processing Toolbox, Computer Vision Toolbox, Parallel Computing Toolbox and Statistics and Machine Learning Toolbox.
We use MatConvNet framework. (As of 2021, we confirmed that the compilation of MatConvNet for CPU works under Windows 10.)

## Installation guidelines
1. `git clone https://github.com/yusukey03012/matObjPose.git`

2. Install MatConvNet from the link or git repo following their instructions:
   
   https://www.vlfeat.org/matconvnet/
   
   https://www.vlfeat.org/matconvnet/install/
   
   https://www.vlfeat.org/matconvnet/quick/
   
3. Download alexnet from below and put it in `data/model`
   
    https://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat

4. Get other dependencies from below and put them into '/external'
- `read_off.m` from:
  
  https://github.com/gpeyre/numerical-tours/tree/master/matlab/toolbox_graph

- `unpackRGBFloat.m` from:
  
  https://rgbd-dataset.cs.washington.edu/software.html

- `minimize_point_to_plane.m` and `get_transform_mat.m` from:
  
  https://jp.mathworks.com/matlabcentral/fileexchange/47152-icp-registration-using-efficient-variants-and-multi-resolution-scheme?s_tid=prof_contriblnk

## Train
To train the model run `runTrainingRotationCNN.m`. 

You need to specify your path to the MatConvNet folder at: `matconvnetpath = /path/to/your/MatConvnet`.
During training it will generate a .mat file that contains the weight of the trained model at each epoch.
You may want to modify your saving directory by changing 'expDir' option of `trainObjectPose.m` and adjust other hyper parameters as well.
We provided a small training dataset in `data/lipton_lemon.mat` with around 1000 images to train our pose detector.  
To train using GPUs, change 'gpus' option, to e.g. [0] or [0,1,2,3].

## Test
To test the trained model run: `testRotationCNN.m`. 

You need to specify your path to the MatConvNet folder at: `matconvnetpath = /path/to/your/MatConvnet`.
You can test the pose detector by setting the path to your trained model: `netPath =/path/to/your/trained/model`.
To test using a GPU, uncommet the lines `%net.move('gpu')` and `%im_= gpuArray(im_);`.  

## Iterative closet point
Run: `performICP.m`.

It runs the point-to-point ICP algorithm first to optimize translation and then perform point-to-plane ICP to optimize both the orientation and translation.
We only provide one 3D model and one point clouds file in this demo but you could try your data by changing
`file_pcd` and  `file_model`. 
We assume here that the region of the object exist is roughly detected in the point clouds.   

### Reference
```
@INPROCEEDINGS{SYG*2019,
  author={K. {Suzui} and Y. {Yoshiyasu} and A. {Gabas} and F. {Kanehiro} and E. {Yoshida}},
  booktitle={2019 IEEE/SICE International Symposium on System Integration (SII)}, 
  title={Toward 6 DOF Object Pose Estimation with Minimum Dataset}, 
  year={2019},
  volume={},
  number={},
  pages={462-467},
  doi={10.1109/SII.2019.8700331}}
```

