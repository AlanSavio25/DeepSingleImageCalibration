"""
Code adapted from Peng-Tao Jiang - github.com/PengtaoJiang
"""

from PIL import Image
import numpy as np
import torch
import argparse
from misc_functions import get_example_params, save_class_activation_images
from calib.calib.datasets.view import read_image, numpy_image_to_torch, resize_image
from torch.nn import ReLU
import torchvision
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
from calib.settings import TRAINING_PATH
from omegaconf import OmegaConf
from tqdm import tqdm
from calib.calib.datasets import get_dataset
from calib.calib.datasets.viz_2d import *
from pathlib import Path
from calib.calib.utils.experiments import load_experiment
import collections
from calib.calib.models.utils_densenet import _DenseLayer, _DenseBlock, _Transition


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        count = 0
        for module_pos, module in self.model.model.features._modules.items():
            x = module(x)  # Forward
            if count == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
            count += 1
        return conv_output, x

    def forward_pass(self, x, head):
        """
            Does a full forward pass on the model
        """
        conv_output, x = self.forward_pass_on_convolutions(x)

        if head == 'roll':
            x = self.model.roll_head(x)
        elif head == 'rho':
            x = self.model.rho_head(x)
        elif head == 'vfov':
            x = self.model.vfov_head(x)
        elif head == 'hfov':
            x = self.model.hfov_head(x)
        elif head == 'k1_hat':
            x = self.model.k1_hat_head(x)
        return conv_output, x


class LayerCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, image_data, target_class=None, head='roll'):
        input_image = image_data['image']

        mean, std = input_image.new_tensor(
            self.mean), input_image.new_tensor(self.std)
        input_image = (input_image - mean[:, None, None]) / std[:, None, None]

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(
            input_image, head)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.model.features.zero_grad()
        if head == 'roll':
            self.model.roll_head.zero_grad()
        elif head == 'rho':
            self.model.rho_head.zero_grad()
        elif head == 'vfov':
            self.model.vfov_head.zero_grad()
        elif head == 'hfov':
            self.model.hfov_head.zero_grad()
        elif head == 'k1_hat':
            self.model.k1_hat_head.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = guided_gradients
        weights[weights < 0] = 0  # discard negative gradients
        # Element-wise multiply the weight with its conv output and then, sum
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS))/255

        return cam


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = '/cluster/home/alpaul/DeepSingleImageCalibration/calib/calib/configs/config_train_aspectratio.yaml'
    experiment = 'exp12_aspectratio_5heads'
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--exp", help="name of experiment")
    argParser.add_argument("-c", "--conf", help="configuration file path")
    argParser.add_argument(
        "--head", help="name of head. Options: roll, rho, vfov, hfov, k1_hat")
    args = argParser.parse_args()
    conf = args.conf
    experiment = args.exp

    output_dir = Path(TRAINING_PATH, experiment)
    conf = OmegaConf.merge(OmegaConf.load(conf), {'train': {'num_workers': 4}})
    data_conf = conf.data
    dataset = get_dataset(data_conf.name)(data_conf)
    test_loader = dataset.get_data_loader('test')
    pretrained_model = load_experiment(experiment, conf.model)

    num_to_viz = 50
    count = 0

    for data in tqdm(test_loader, desc='Visualizing LayerCAM', ascii=True, disable=False):
        if not os.path.exists(f'results/{args.head}'):
            os.makedirs(f'results/{args.head}')
        file_name_to_export = f'{args.head}/layercam'+data['path'][0].split('/')[-1]
        count += 1

        layer_cam = LayerCam(pretrained_model, target_layer=6)
        cam = layer_cam.generate_cam(data, target_class=None, head=args.head)
        save_class_activation_images(Image.fromarray(resize_image(read_image(Path(
            data['path'][0]), grayscale=False), resize_method='simple')), cam, file_name_to_export)

        if count >= num_to_viz:
            break
    print('Layer cam completed')
