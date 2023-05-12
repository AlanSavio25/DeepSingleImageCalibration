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
import pycolmap
import tqdm
import pandas as pd
import os


def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(prog='Deep Single Image calibration',
                                     description='Inference code to show the predicted camera parameters')
    parser.add_argument('--img_dir', default='images/')
    args = parser.parse_args()

    if not os.path.exists('./weights/checkpoint_best.tar'):
        os.system('mkdir -p weights')
        logger.info("Downloading weights into ./weights/")
        torch.hub.download_url_to_file('https://github.com/AlanSavio25/DeepSingleImageCalibration/releases/download/v1/checkpoint_best.tar',
                                       'weights/checkpoint_best.tar', hash_prefix='a84cb9606931529bab33524b15cbfd7370b4d7593e2849b3f1dac0b9b3dd2583')
    os.system('mkdir -p ./results/')
    results_dir = './results/'

    experiment = 'weights'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    roll_centers, rho_centers, fov_centers, k1_hat_centers = get_bin_centers(
        device)
    conf = 'calib/calib/configs/config_test.yaml'
    logger.info(f'Starting test {experiment}')
    conf = OmegaConf.merge(OmegaConf.load(
        conf), {'data': {'data_dir': args.img_dir}})
    dataset = get_dataset(conf.data.name)(conf.data)
    test_loader = dataset.get_data_loader('test')
    model = load_experiment(experiment, conf.model).eval()

    results = []

    for data in tqdm.tqdm(test_loader, desc='Testing', ascii=True, disable=False):
        pred = model(data)
        num_bins = 256

        image_path = data['path'][0]
        im = read_image(image_path, grayscale=False)
        h, w, _ = im.shape

        roll_centers = torch.linspace(-45.0,
                                      45.0+(90./(num_bins-1)), num_bins+1)
        rho_centers = torch.linspace(-1., 1.+(2./(num_bins-1)), num_bins+1)
        fov_centers = torch.linspace(20., 105.+(85./(num_bins-1)), num_bins+1)
        k1_hat_centers = torch.linspace(-0.45,
                                        0.+(0.45/(num_bins-1)), num_bins+1)
        roll_edges = (roll_centers - ((roll_centers[1] - roll_centers[0])/2.))
        rho_edges = (rho_centers - ((rho_centers[1] - rho_centers[0])/2.))
        fov_edges = (fov_centers - ((fov_centers[1] - fov_centers[0])/2.))
        k1_hat_edges = (k1_hat_centers -
                        ((k1_hat_centers[1] - k1_hat_centers[0])/2.))

        roll = (roll_centers[pred['roll'].argmax(1)])
        rho = ((rho_centers[pred['rho'].argmax(1)]) * 0.35)
        vfov = (fov_centers[pred['vfov'].argmax(1)])
        fy_px = 1 / (torch.tan(vfov * (torch.pi/180.) / 2) * 2 / h)
        k1_hat = (k1_hat_centers[pred['k1_hat'].argmax(1)])
        k1 = k1_hat * (fy_px/h)**2

        # Compute pitch from predicted rho
        u0 = h / 2.
        v0 = w / 2.
        rho_px = rho.cpu().item() * h

        img_pts = [u0, rho_px + v0]
        camera = pycolmap.Camera(
            model='RADIAL',
            width=w,
            height=h,
            params=[fy_px, u0, v0, k1, 0.],
        )
        normalized_coords = np.array(camera.image_to_world(img_pts))
        camera_no_distortion = pycolmap.Camera(
            model='RADIAL',
            width=w,
            height=h,
            params=[fy_px, u0, v0, 0.0, 0.0],
        )
        back_to_image = np.array(
            camera_no_distortion.world_to_image(normalized_coords))
        tau = (back_to_image[1] - v0) / h

        pitch = np.arctan(tau/(fy_px/h))


        output = {'pred_roll': roll,
                       'pred_rho': rho,
                       'pitch': pitch,
                       'pred_fov': vfov,
                       'focal_length_pixels': fy_px,
                       'k1': k1,
                       'pred_k1_hat': k1_hat
                 }

        plt = plot_row([{**{k: output[k].unsqueeze(0).cpu().item()
                          for k in output}, 'image': data['image'].numpy().squeeze(0).transpose((1, 2, 0)), 'path': [image_path]}], pred_annotate=[
                     'roll', 'rho', 'fov', 'k1_hat'])

        plt.savefig(results_dir + image_path.split('/')[-1])

        pd.DataFrame(output).T.to_csv(Path(results_dir, str(
            Path(image_path).stem)+'.csv'), header=False, index=True)

    logger.info(f"Saved results into {results_dir}")
    
if __name__ == '__main__':
    main()
