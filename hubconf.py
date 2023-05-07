dependencies = ['torch', 'os', 'cv2', 'pycolmap', 'numpy']
import torch
from calib.calib.utils.experiments import load_experiment
import os
import cv2
import pycolmap
import numpy as np


def calib(image_path=None):
    """
    DenseNet-161 based Calibration model
    image_path (str): if not None, image is loaded and passed through the network. Default: None.
    return: output dict {'model': model, **predictions}
    """
    output = {}
    if not os.path.exists('./weights/checkpoint_best.tar'):
        os.system('mkdir -p weights')
        torch.hub.download_url_to_file('https://github.com/AlanSavio25/DeepSingleImageCalibration/releases/download/v1/checkpoint_best.tar', 
                'weights/checkpoint_best.tar', hash_prefix='a84cb9606931529bab33524b15cbfd7370b4d7593e2849b3f1dac0b9b3dd2583')
    model = load_experiment('weights', {'name': 'densenet'})

    if image_path is not None:

        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
        im = im.transpose((2, 0, 1))  # HxWxC to CxHxW
        im = torch.from_numpy(im / 255.).float().unsqueeze(0)
        pred = model({'image': im})
        
        num_bins = 256
        roll_centers = torch.linspace(-45.0, 45.0+(90./(num_bins-1)), num_bins+1)
        rho_centers = torch.linspace(-1., 1.+(2./(num_bins-1)), num_bins+1)
        fov_centers = torch.linspace(20., 105.+(85./(num_bins-1)), num_bins+1)
        k1_hat_centers = torch.linspace(-0.45, 0.+(0.45/(num_bins-1)), num_bins+1)
        roll_edges = (roll_centers - ((roll_centers[1] - roll_centers[0])/2.))
        rho_edges = (rho_centers - ((rho_centers[1] - rho_centers[0])/2.))
        fov_edges = (fov_centers - ((fov_centers[1] - fov_centers[0])/2.))
        k1_hat_edges = (k1_hat_centers - ((k1_hat_centers[1] - k1_hat_centers[0])/2.))

        roll = (roll_centers[pred['roll'].argmax(1)])
        rho = ((rho_centers[pred['rho'].argmax(1)]) * 0.35)
        vfov = (fov_centers[pred['vfov'].argmax(1)])
        fy_px = 1 / (torch.tan(vfov * (torch.pi/180.) / 2) * 2 / h)
        k1_hat = (k1_hat_centers[pred['k1_hat'].argmax(1)])
        k1 = k1_hat * (fy_px/h)**2

        # Compute pitch from predicted rho
        u0 = h / 2.
        v0 = w / 2.
        rho_px = rho * h

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

        output.update({'roll_degrees': roll,
            'distorted_offset_fractionofheight': rho,
            'pitch': pitch,
            'vfov_degrees': vfov,
            'focal_length_pixels': fy_px,
            'k1': k1,
            'k1_hat': k1_hat
            })

    return model, output


if __name__=='__main__':
    print(calib(image_path='images/video1-00150.jpg'))
