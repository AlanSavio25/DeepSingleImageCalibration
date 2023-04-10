import torch

model = torch.hub.load('AlanSavio25/DeepSingleImageCalibration', 'calib', pretrained=True).eval()

im = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
im = numpy_image_to_torch(im)

assert im.ndim == 3
im = im.transpose((2, 0, 1))  # HxWxC to CxHxW
im = torch.from_numpy(im / 255.).float()
pred = model({'image': im})