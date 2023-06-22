from calib.demo import DeepCalibration


def calibrator(**kwargs):
    return DeepCalibration(**kwargs)


dependencies = ['torch', 'cv2', 'pycolmap', 'numpy']
