import numpy as np
import imageio
import math as m
from calib.calib.datasets.interpol import *
from numpy.lib.scimath import sqrt as csqrt
import pycolmap


def deg2rad(deg):
    return deg*m.pi/180


def crop_distortion(image360_path, f, H, W, az, el, roll, k1, k2):

    u0 = W / 2.
    v0 = H / 2.
    grid_x, grid_y = np.meshgrid(list(range(W)), list(range(H)))

    if isinstance(image360_path, str):
        image360 = imageio.imread(image360_path)
    else:
        image360 = image360_path.copy()

    ImPano_W = np.shape(image360)[1]
    ImPano_H = np.shape(image360)[0]
    x_ref = 1
    y_ref = 1

    # 1. Projection on the camera plane

    img_pts = np.array(list(zip(grid_x.ravel(), grid_y.ravel())))
    camera = pycolmap.Camera(
        model='RADIAL',
        width=W,
        height=H,
        params=[f, u0, v0, k1, 0],
    )
    normalized_coords = np.array(camera.image_to_world(img_pts))
    X_Cam = normalized_coords[:, 0].reshape((H, W))
    Y_Cam = -normalized_coords[:, 1].reshape((H, W))

    # 2. Projection on the sphere

    AuxVal = np.multiply(X_Cam, X_Cam) + np.multiply(Y_Cam, Y_Cam)
    alpha_cam = np.real(csqrt(1 + AuxVal))
    alpha_div = AuxVal + 1
    alpha_cam_div = np.divide(alpha_cam, alpha_div)
    X_Sph = np.multiply(X_Cam, alpha_cam_div)
    Y_Sph = np.multiply(Y_Cam, alpha_cam_div)
    Z_Sph = alpha_cam_div  # - k1

    # 3. Rotation of the sphere

    coords = np.vstack((X_Sph.ravel(), Y_Sph.ravel(), Z_Sph.ravel()))
    rot_el = np.array([1., 0., 0., 0., np.cos(deg2rad(el)), -np.sin(deg2rad(el)),
                      0., np.sin(deg2rad(el)), np.cos(deg2rad(el))]).reshape((3, 3))
    rot_az = np.array([np.cos(deg2rad(az)), 0., np.sin(deg2rad(
        az)), 0., 1., 0., -np.sin(deg2rad(az)), 0., np.cos(deg2rad(az))]).reshape((3, 3))
    rot_roll = np.array([np.cos(deg2rad(roll)), -np.sin(deg2rad(roll)), 0., np.sin(
        deg2rad(roll)), np.cos(deg2rad(roll)), 0., 0., 0., 1.]).reshape((3, 3))
    sph = rot_roll.dot(rot_el.dot(coords))
    sph = rot_az.dot(sph)
    sph = sph.reshape((3, H, W)).transpose((1, 2, 0))
    X_Sph, Y_Sph, Z_Sph = sph[:, :, 0], sph[:, :, 1], sph[:, :, 2]

    # 4. cart 2 sph
    ntheta = np.arctan2(X_Sph, Z_Sph)
    nphi = np.arctan2(Y_Sph, np.sqrt(Z_Sph**2 + X_Sph**2))
    pi = m.pi

    # 5. Sphere to pano
    min_theta = -pi
    max_theta = pi
    min_phi = -pi / 2.
    max_phi = pi / 2.

    min_x = 0
    max_x = ImPano_W - 1.0
    min_y = 0
    max_y = ImPano_H - 1.0

    # for x
    a = (max_theta - min_theta) / (max_x - min_x)
    b = max_theta - a * max_x  # from y=ax+b %% -a;
    nx = (1. / a) * (ntheta - b)

    # for y
    a = (min_phi - max_phi) / (max_y - min_y)
    b = max_phi - a * min_y  # from y=ax+b %% -a;
    ny = (1. / a) * (nphi - b)

    # 6. Final step interpolation and mapping

    im = np.array(interp2linear(image360, nx, ny), dtype=np.uint8)
    del image360

    return im
