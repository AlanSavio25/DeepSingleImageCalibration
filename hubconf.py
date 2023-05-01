dependencies = ['torch', 'os']
import torch
from calib.calib.utils.experiments import load_experiment
import os


def calib():
    if not os.path.exists('./weights/checkpoint_best.tar'):
        os.system('mkdir -p weights')
        torch.hub.download_url_to_file('https://github.com/AlanSavio25/DeepSingleImageCalibration/releases/download/v1/checkpoint_best.tar', 
                'weights/checkpoint_best.tar', hash_prefix='a84cb9606931529bab33524b15cbfd7370b4d7593e2849b3f1dac0b9b3dd2583')
    experiment = 'weights'
    model = load_experiment(experiment, {'name': 'densenet'})
    return model
