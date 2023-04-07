from pathlib import Path
import numpy as np
import cv2
# TODO: consider using PIL instead of OpenCV as it is heavy and only used here
import torch

# from ..geometry import Camera, Pose


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image


def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def crop(image, size, *, random=True, other=None, camera=None,
         return_bbox=False, centroid=None):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    if random:
        top = np.random.randint(0, h - h_new + 1)
        left = np.random.randint(0, w - w_new + 1)
    elif centroid is not None:
        x, y = centroid
        top = np.clip(int(y) - h_new // 2, 0, h - h_new)
        left = np.clip(int(x) - w_new // 2, 0, w - w_new)
    else:
        top = left = 0

    image = image[top:top+h_new, left:left+w_new]
    ret = [image]
    if other is not None:
        ret += [other[top:top+h_new, left:left+w_new]]
    if camera is not None:
        ret += [camera.crop((left, top), (w_new, h_new))]
    if return_bbox:
        ret += [(top, top+h_new, left, left+w_new)]
    return ret


def zero_pad(size, *images):
    ret = []
    for image in images:
        h, w = image.shape[:2]
        padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret


def resize_image(im, resize_method='simple', width=224, height=224):
    if resize_method == 'simple':
        new_img = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    elif resize_method == 'letterbox':
        ih, iw, _ = im.shape
        eh, ew = 224, 224  # expected size
        scale = min(eh / ih, ew / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)
        image = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        new_img = np.full((eh, ew, 3), 128, dtype='uint8')
        # fill new image with the resized image and centered it
        new_img[(eh - nh) // 2:(eh - nh) // 2 + nh,
                (ew - nw) // 2:(ew - nw) // 2 + nw,
                :] = im.copy()
    return new_img
