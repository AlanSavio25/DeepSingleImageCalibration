import torch

_, results = torch.hub.load('AlanSavio25/DeepSingleImageCalibration', 'calib', image_path='images/video1-00150.jpg', force_reload=True)
print(results)
