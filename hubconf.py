dependencies = ['torch', 'os']
import torch
from calib.calib.utils.experiments import load_experiment
import os


def calib(pretrained=True):
    os.system('mkdir weights')
    os.system('wget https://github.com/AlanSavio25/DeepSingleImageCalibration/releases/download/v1/checkpoint_best.tar -P weights') 
    experiment = 'weights'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_experiment(experiment, {'name': 'densenet'}).eval()
    return model
