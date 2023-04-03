# Deep Single Image Calibration
In this project, we train a neural network to estimate from a single image its roll, pitch (parameterized by offset of horizon from image centre), focal length (parameterized by field of view), and radial distortion parameter k1 (parameterized by k1_hat where k1 and focal length are decoupled).
This work is closest to the paper [Deep Single Image Camera Calibration With Radial Distortion](https://openaccess.thecvf.com/content_CVPR_2019/html/Lopez_Deep_Single_Image_Camera_Calibration_With_Radial_Distortion_CVPR_2019_paper.html) and 
[DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras](https://dl.acm.org/doi/10.1145/3278471.3278479)



## Installation

This project requires Python>=3.6 and PyTorch>1.1. The following steps will install the `calib` package using `setup.py` and `requirements.txt`. 
```
git clone git@github.com:AlanSavio25/DeepSingleImageCalibration.git
cd DeepSingleImageCalibration
pip install -e .
```

## Testing [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aqw2NQZsR7PP-rN55G7s9kI6shv1Qxtj)

In the above Colab page, you can add your images and test the network. The Colab page has access to 2 trained weights (more info below). 
Currently, for images that are not 224x224 in size, we center crop the image before passing it through the network. In the near future, we will try to use all of the image content.

## Trained Network Weights

1. [Network 1](https://drive.google.com/drive/folders/1DKH6sJBr1WJlUo2kjhpTb8JddwyymcJB) is trained to estimate 3 parameters: roll, rho, field of view.

2. [Network 2](https://drive.google.com/drive/folders/1DKH6sJBr1WJlUo2kjhpTb8JddwyymcJB) is trained to estimate 4 parameters: roll, rho, field of view, k1_hat.

Network 1 has been tested more and works well.

## Visualization of performance

## Dataset used for training the network

[Insert Size of Train Test Val + Sampling range for each parameter]

### Parameter distribution

[Insert distribution of each parameter here]

### Example dataset images


## Training

### Dataset Generation Pipeline


### Training Experiment
