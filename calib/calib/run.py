import argparse
from calib import logger
from pathlib import Path
import torch
import numpy as np
from calib.calib.datasets import get_dataset
from omegaconf import OmegaConf
from calib.calib.utils.experiments import load_experiment
from calib.calib.utils.helper_functions import *
from calib.calib.datasets.viz_2d import *
import tqdm
import pandas as pd
import os


def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(prog='Deep Single Image calibration',
                                     description='Inference code to show the predicted camera parameters')
    parser.add_argument('--img_dir', default='images/')
    parser.add_argument('--weights_dir', default='weights/')
    args = parser.parse_args()

    os.system('mkdir -p ./results/')
    data_dir = args.img_dir
    results_dir = './results/'

    experiment = args.weights_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    roll_centers, rho_centers, fov_centers, k1_hat_centers = get_bin_centers(
        device)
    conf = 'calib/calib/configs/config_test.yaml'
    logger.info(f'Starting test {experiment}')
    conf = OmegaConf.merge(OmegaConf.load(
        conf), {'data': {'data_dir': data_dir}})
    dataset = get_dataset(conf.data.name)(conf.data)
    test_loader = dataset.get_data_loader('test')
    model = load_experiment(experiment, conf.model).eval()

    results = []

    for data in tqdm.tqdm(test_loader, desc='Testing', ascii=True, disable=False):
        pred = model(data)
        output = pred['roll']
        entropy = compute_categorical_entropy(output)
        pred['roll_entropy'] = entropy
        pred_class = output.argmax(1)
        pred_deg = torch.tensor(
            roll_centers[pred_class], dtype=torch.float64, device=device)
        pred['roll'] = pred_deg

        output = pred['rho']
        entropy = compute_categorical_entropy(output)
        pred['rho_entropy'] = entropy
        pred_class = output.argmax(1)
        pred_norm_rho = torch.tensor(
            rho_centers[pred_class], dtype=torch.float64, device=device)
        pred_ratio_rho = pred_norm_rho * 0.35
        pred['rho'] = pred_ratio_rho

        output = pred['vfov']
        entropy = compute_categorical_entropy(output)
        pred['vfov_entropy'] = entropy
        pred_class = output.argmax(1)
        pred_fov_deg = torch.tensor(
            fov_centers[pred_class], dtype=torch.float64, device=device)
        pred['fov'] = pred_fov_deg
        del pred['vfov']

        output = pred['k1_hat']
        entropy = compute_categorical_entropy(output)
        pred['k1_hat_entropy'] = entropy
        pred_class = output.argmax(1)
        pred_k1_hat = torch.tensor(
            k1_hat_centers[pred_class], dtype=torch.float64, device=device)
        pred['k1_hat'] = pred_k1_hat
        result = {'path': data['path'],
                  **{'pred_'+str(pred_key): pred[pred_key].unsqueeze(0).cpu().item()
                      if isinstance(pred[pred_key], torch.Tensor)
                      else pred[pred_key]
                      for pred_key in pred}
                  }
        results.append(
            {**result, 'image': data['image'].numpy().squeeze(0).transpose((1, 2, 0))})
        pd.DataFrame(result).T.to_csv(Path(results_dir, str(
            Path(data['path'][0]).stem)+'.csv'), header=False, index=True)
    for i in range(len(results)):
        p = plot_row([results[i]], pred_annotate=[
                     'roll', 'rho', 'fov', 'k1_hat'], titles=[])
        p.savefig(results_dir + results[i]['path'][0].split('/')[-1])


if __name__ == '__main__':
    main()
