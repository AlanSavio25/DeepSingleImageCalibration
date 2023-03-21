# Deep Single Image Calibration

In this project, we train a neural network to estimate from a single image its roll, tilt (parameterized by offset of horizon from image centre), focal length (parameterized by vertical/horizontal field of view), and radial distortion parameter k1 (parameterized by k1_hat where k1 and focal length are decoupled).

This work is closest to the paper [Deep Single Image Camera Calibration With Radial Distortion](https://openaccess.thecvf.com/content_CVPR_2019/html/Lopez_Deep_Single_Image_Camera_Calibration_With_Radial_Distortion_CVPR_2019_paper.html) and 
[DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras](https://dl.acm.org/doi/10.1145/3278471.3278479)

## Installation

This project requires Python>=3.6 and PyTorch>1.1. The following steps will install the `calib` package using `setup.py` and `requirements.txt`. 

```
git clone git@github.com:AlanSavio25/DeepSingleImageCalibration.git
cd DeepSingleImageCalibration
pip install -e .
```

## Training

### Dataset

We used the [SUN360](https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM) dataset to obtain 360° panoramas, from which we generated 274,072 images. We use a train-val-test split of 264,072 - 5000 - 2000. To generate these images, first combine all SUN360 images into one directory `SUN360/total/`, then run:

```
cd calib/calib/datasets/
python image_generation.py
```
This varies the roll, tilt, yaw, field of view, distortion and aspect ratio to generate multiple images from each panorama.

Then, run the following script to create the train-val-test split:

```
mkdir -p split
python create_split.py
```

#### Examples of Images in the generated dataset
<img width="500" alt="dataset2_examples" src="https://user-images.githubusercontent.com/30126243/226637738-0fa8b885-07e0-457e-95f1-c0668ade03c5.png">

#### Distributions of parameters in the dataset
<img width="500" alt="Screenshot 2023-03-21 at 15 30 19" src="https://user-images.githubusercontent.com/30126243/226638352-d9ebf5c9-e9f2-4848-a710-a2b82393f1bd.png">


### Training experiment


Our framework is derived from [pixloc](https://github.com/cvg/pixloc/tree/master/pixloc/pixlib), where you will find more information about training.

To start a training experiment, run:

```
python -m calib.calib.train experiment_name --conf calib/calib/configs/config_train.yaml
```

It creates a new directory `experiment_name/` in TRAINING_PATH (set inside settings.py) and saves the config, model checkpoints, logs of stdout, and Tensorboard summaries.

`--overfit` flag loops the training and validation sets on a single batch (useful to test losses and metrics).

*Monitoring the training*: Launch a Tensorboard session with `tensorboard --logdir=path/to/TRAINING_PATH` to visualize losses and metrics, and compare them across experiments.

## Testing [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aqw2NQZsR7PP-rN55G7s9kI6shv1Qxtj)

In the above Colab page, you can add your images and test the network. The Colab page has access to the 4-headed network's trained weights.

For images that are not square, we resize them to a square before passing it through the network.

## Trained Network Weights

1. [Network 1](https://drive.google.com/drive/folders/1DKH6sJBr1WJlUo2kjhpTb8JddwyymcJB) is trained to estimate 3 parameters: roll, rho, field of view.

2. [Network 2](https://drive.google.com/drive/folders/1DKH6sJBr1WJlUo2kjhpTb8JddwyymcJB) is trained to estimate 4 parameters: roll, rho, field of view, k1_hat.

3. [Network 3](TODO) is trained to estimate 4 parameters: same as above, but trained on images with varying aspect ratios (1 and 2 are trained on square images only).


## Interpreting the neural network
We also add a notebook (visualize_layercam.ipynb) to visualize the gradients of the network. Before using this, run the following:
```
cd class_activation_maps
python layercam.py -c path/to/config_train.yaml --head roll -e exp12_aspectratio_5heads
```
We adapt code from [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations).

<img width="500" alt="Screenshot 2023-03-21 at 17 07 10" src="https://user-images.githubusercontent.com/30126243/226669347-a263b86b-d76e-4ca5-b2a9-37746880f5ef.png">

Notice that the network focuses on straight lines and horizons.
