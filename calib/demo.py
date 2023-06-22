from pathlib import Path

import numpy as np
import torch

from calib.calib.datasets.view import (
    read_image, numpy_image_to_torch, resize_image)
from calib.calib.datasets.viz_2d import plot_row
from calib.calib.utils.experiments import load_experiment

try:
    import pycolmap
except ImportError:
    pycolmap = None


CHECKPOINT_URL = 'https://github.com/AlanSavio25/DeepSingleImageCalibration/releases/download/v1/checkpoint_best.tar'
CHECKPOINT_HASH = 'a84cb9606931529bab33524b15cbfd7370b4d7593e2849b3f1dac0b9b3dd2583'


def adjust_rho_distortion(rho, fy_px, k1, w, h):
    if pycolmap is None:
        raise ImportError("pycolmap is required to handle distorted images.")
    cx = w / 2
    cy = h / 2
    image_points = [cx, rho * h + cy]
    camera = pycolmap.Camera(
        model='RADIAL',
        width=w,
        height=h,
        params=[fy_px, cx, cy, k1, 0.],
    )
    normalized_coords = np.array(camera.image_to_world(image_points))
    camera_no_distortion = pycolmap.Camera(
        model='RADIAL',
        width=w,
        height=h,
        params=[fy_px, cx, cy, 0.0, 0.0],
    )
    reprojected_points = np.array(
        camera_no_distortion.world_to_image(normalized_coords))
    tau = (reprojected_points[1] - cy) / h
    return tau


class DeepCalibration:
    def __init__(self, device=None):
        checkpoint_path = Path(
            torch.hub.get_dir(), 'deepcalib', 'checkpoint_best.tar')
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(
                CHECKPOINT_URL, checkpoint_path, hash_prefix=CHECKPOINT_HASH)
        model = load_experiment(checkpoint_path.parent, {'name': 'densenet'})
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device).eval()
        self.device = device

    def calibrate(self, image, force_pinhole=True):
        h, w, _ = image.shape
        image = numpy_image_to_torch(resize_image(image)).unsqueeze(0)
        pred = self.model({'image': image.to(self.device)})

        num_bins = 256
        roll_centers = torch.linspace(-45.0,
                                      45.0+(90./(num_bins-1)), num_bins+1)
        rho_centers = torch.linspace(-1., 1.+(2./(num_bins-1)), num_bins+1)
        fov_centers = torch.linspace(20., 105.+(85./(num_bins-1)), num_bins+1)

        roll = (roll_centers[pred['roll'].argmax(1)])
        rho = ((rho_centers[pred['rho'].argmax(1)]) * 0.35)
        vfov = (fov_centers[pred['vfov'].argmax(1)])
        fy_px = h / 2 / torch.tan(vfov * (torch.pi/180.) / 2)

        roll, rho, vfov, fy_px = [x.cpu().item() for x in [roll, rho, vfov, fy_px]]
        ret = dict(
           roll=roll,
           rho=rho,
           vertical_fov=vfov,
           focal_length_pixels=fy_px,
        )

        if not force_pinhole:
            k1_hat_centers = torch.linspace(
                -0.45, 0.+(0.45/(num_bins-1)), num_bins+1)
            k1_hat = (k1_hat_centers[pred['k1_hat'].argmax(1)])
            k1 = k1_hat * (fy_px/h)**2
            ret.update(dict(
               k1=k1.cpu().item(),
               pred_k1_hat=k1_hat.cpu().item(),
            ))
            rho = adjust_rho_distortion(rho, fy_px, k1, w, h)
        ret['pitch'] = np.arctan(rho * h / fy_px)
        return ret

    def calibrate_from_path(self, image_path):
        image = read_image(image_path, grayscale=False)
        return self.calibrate(image)

    def visualize(self, image, ret, **kwargs):
        plot_row(
            [{**ret, 'image': image}],
            pred_annotate=['roll', 'rho', 'fov', 'k1_hat'],
            **kwargs,
        )
