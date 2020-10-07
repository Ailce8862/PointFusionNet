# PointFusionNet: Point feature fusion network for 3D point clouds analysis
This repository contains the author's implementation in Pytorch for the paper: PointFusionNet: Point feature fusion network for 3D point clouds analysis 
## Requirement
* Ubuntu 16.04
* Python 3 (recommend Anaconda3)
* Pytorch 0.3/0.4
* CMake > 5.4
* CUDA 9.0 + cuDNN 7.6
## Download Dataset
### Shape Classification
Download and unzip ModelNet40 (415M). Replace $data_root$ in cfgs/config_*_cls.yaml with the dataset parent path.
### ShapeNet Part Segmentation
Download and unzip ShapeNet Part (674M). Replace $data_root$ in cfgs/config_*_partseg.yaml with the dataset path.
## Train ModelNet40
Run train_msg_cls.py
## Train ShapeNet Part
Run train_ssn_partseg.py
