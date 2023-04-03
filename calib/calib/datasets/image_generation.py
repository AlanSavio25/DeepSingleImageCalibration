import numpy as np
from calib.calib.datasets.spherical_distortion import *
from glob import glob
from imageio import imread, imsave
from scipy.stats import cauchy, lognorm, norm, uniform, truncnorm
import json
from tqdm import tqdm
import os
import random
from time import sleep, time
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from numpy.random import seed

# random.seed(1)
# np.random.seed(1)

INPUT_DIR = "./SUN360/total/"
OUTPUT_DIR = "./calibration_dataset/"

all_panoramas = glob(INPUT_DIR + "*", recursive=False)
# selected_panoramas = np.random.choice(all_panoramas, size=250, replace=False)

num_samples = 8  # per pano
ranges = {
    "f": [0.4, 1.451],  # unit: ratio [focal length (px) / image_width (px)]
    "roll": [-45 * np.pi / 180., 45 * np.pi / 180.],
    "yaw": [0, 2 * np.pi],
    "tau": [0.15, 0.85],
    "k1": [-0.3, 0.],
    "aspect_ratio": [np.log(9/16), np.log(16/9)] # log to make it symmetric around aspect ratio 1:1
}

def max_radius(a, b):
    discrim = a * a - 4 * b
    if np.isfinite(discrim) and discrim >= 0.0:
        discrim = np.sqrt(discrim) - a
        if discrim > 0.0:
            return 2.0 / discrim
    return np.inf


def brown_max_radius(k1, k2):
    # fold the constants from the derivative into a and b
    a = k1 * 3
    b = k2 * 5
    return np.sqrt(max_radius(a, b))


def image_generator(pano):
    seed()

    # Get aspect ratios
    aspect_ratios = np.exp(np.random.uniform(low=ranges["aspect_ratio"][0], high=ranges["aspect_ratio"][1], size=num_samples))
    heights = np.empty(num_samples)
    widths = np.empty(num_samples)
    # select height and width such that smaller side = 224
    for i in range(num_samples):
        if aspect_ratios[i] >= 1.:  # landscape
            heights[i] = 224
            widths[i] = 224 * aspect_ratios[i]
        else:  # portrait
            widths[i] = 224
            heights[i] = 224 * (1/aspect_ratios[i])
    heights = np.around(heights, decimals=0)
    widths = np.around(widths, decimals=0)

    # Get yaws
    start = ranges["yaw"][0]
    end = ranges["yaw"][1]
    step = (end - start) / num_samples
    yaw = np.arange(start, end, step)

    # Sample roll - truncnorm
    myclip_a, myclip_b = ranges["roll"]
    my_mean = 0
    my_std = 0.65

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    roll = truncnorm.rvs(a, b, loc=[my_mean] *
                         num_samples, scale=[my_std] * num_samples)

    # Sample k1

    myclip_a, myclip_b = ranges['k1']
    my_mean = 0.0
    my_std = 0.125
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    k1 = truncnorm.rvs(a, b, loc=[my_mean] *
                       num_samples, scale=[my_std] * num_samples)

    # Sample focal_length_ratio (based on k1)

    myclip_a, myclip_b = ranges["f"]
    my_mean = 0.6
    my_std = 0.3
    assert len(k1) == num_samples
    f_ratio_height = np.empty(num_samples)
    f_ratio_width = np.empty(num_samples)
    f_px = np.empty(num_samples)
    vfov = np.empty(num_samples)
    hfov = np.empty(num_samples)
    for i in range(num_samples):
        min_permissible_rmax = np.sqrt(
            (heights[i] / 2)**2 + (widths[i] / 2)**2)  # distance to image corner
        r_max = brown_max_radius(k1=k1[i], k2=0)
        # r_max_im = f_px * r_max * (1 + k1*r_max**2)
        # function of r_max_im: f_px = r_max_im / (r_max * (1 + k1*r_max**2))
        lowest_possible_f_px = min_permissible_rmax / \
            (r_max * (1 + k1[i] * r_max**2))
        if aspect_ratios[i] >= 1.: # landscape, so sample hfov and scale vfov using ar
            lowest_possible_f_ratio = lowest_possible_f_px / widths[i]
            lowest_a, a, b = (lowest_possible_f_ratio - my_mean) / \
                my_std, (myclip_a - my_mean) / \
                my_std, (myclip_b - my_mean) / my_std
            f_ratio_width[i] = truncnorm.rvs(max(lowest_a, a), b,
                                              loc=my_mean, scale=my_std)
            f_px[i] = f_ratio_width[i] * widths[i]
            hfov[i] = 2 * np.arctan(1 / (2 * f_ratio_width[i]))
            
            f_ratio_height[i] = f_px[i] / heights[i]
            vfov[i] = 2 * np.arctan(1 / (2 * f_ratio_height[i]))

            
        else: # portrait
            lowest_possible_f_ratio = lowest_possible_f_px / heights[i]
            lowest_a, a, b = (lowest_possible_f_ratio - my_mean) / \
                my_std, (myclip_a - my_mean) / \
                my_std, (myclip_b - my_mean) / my_std
            f_ratio_height[i] = truncnorm.rvs(max(lowest_a, a), b,
                                              loc=my_mean, scale=my_std)
            f_px[i] = f_ratio_height[i] * heights[i]
            vfov[i] = 2 * np.arctan(1 / (2 * f_ratio_height[i]))
            f_ratio_width[i] = f_px[i] / widths[i]
            hfov[i] = 2 * np.arctan(1 / (2 * f_ratio_width[i]))
            
            assert f_ratio_height[i] >= lowest_possible_f_ratio, \
            f'{f_ratio_height[i], lowest_possible_f_ratio, max(lowest_possible_f_ratio, myclip_a), }'
            assert myclip_a <= f_ratio_height[i] <= myclip_b

        np.testing.assert_almost_equal(
            vfov[i], 2 * np.arctan((heights[i] * np.tan(hfov[i]/2))/widths[i]))
        
    assert len(f_ratio_height) == num_samples

    # Setting k1_hat after k1 and f_ratio_height have been sampled
    k1_hat = k1 / (f_ratio_height**2)

    # Sample tau

    myclip_a, myclip_b = ranges["tau"]
    my_mean = 0.5
    my_std = 0.3
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    sampled_horizon = truncnorm.rvs(a, b, loc=[my_mean] * num_samples,
                                    scale=[my_std] * num_samples)  # fraction of image height
    tau = sampled_horizon - 0.5
    pitch = np.arctan(tau / f_ratio_height)

    r = (tau * heights) / f_px
    d = (1 + k1 * r**2)
    rho = f_ratio_height * d * r
    heights = heights.astype(np.int)
    widths = widths.astype(np.int)

    for i in range(num_samples):
        crop_name = pano.split('/')[-1][:-4] + '_' + str(i).zfill(2)
        try:
            crop = crop_distortion(pano, f=f_px[i], H=heights[i], W=widths[i],
                                   az=yaw[i] * 180 / np.pi, el=pitch[i] * 180 / np.pi,
                                   roll=roll[i] * 180 / np.pi, k1=k1[i], k2=0)
        except ValueError as e:
            print(f"Value Error: {e}")
            print(pano)
            continue
        labels = {
            'name': crop_name + '.jpg',
            'k1_hat': k1_hat[i],
            'vfov': vfov[i],
            'hfov': hfov[i],
            'roll': roll[i],
            'rho': rho[i],
            'tau': tau[i],
            'k1': k1[i],
            'k2': 0,
            'aspect_ratio': aspect_ratios[i],
            'pitch': pitch[i],
            'focal_length_ratio_height': f_ratio_height[i],
            'focal_length_ratio_width': f_ratio_width[i],
            'f_px': f_px[i],
            'height': np.float(heights[i]),
            'width': np.float(widths[i]),
            'yaw': yaw[i],

        }
        imsave(f'{OUTPUT_DIR}/{crop_name}.jpg', crop)
        with open(f'{OUTPUT_DIR}/{crop_name}.json', 'w+') as flhd:
            json.dump(labels, flhd)


start_time = time()
num_jobs = 16
print("Total cpu count: ", int(cpu_count()))
print("Number of jobs: ", num_jobs)


Parallel(n_jobs=num_jobs)(delayed(image_generator)(pano)
                          for pano in all_panoramas)
parallel_execution_time = time() - start_time
print(parallel_execution_time)
