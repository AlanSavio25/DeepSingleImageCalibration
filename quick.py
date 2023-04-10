import torch
import cv2

model = torch.hub.load('AlanSavio25/DeepSingleImageCalibration', 'calib', pretrained=True).eval()

im = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
im = im.transpose((2, 0, 1))  # HxWxC to CxHxW
im = torch.from_numpy(im / 255.).float().unsqueeze(0)
pred = model({'image': im})


num_bins = 256

roll_centers = torch.linspace(-45.0, 45.0+(90./(num_bins-1)), num_bins+1)
rho_centers = torch.linspace(-1., 1.+(2./(num_bins-1)), num_bins+1)
fov_centers = torch.linspace(20., 105.+(85./(num_bins-1)), num_bins+1)
k1_hat_centers = torch.linspace(-0.45, 0.+(0.45/(num_bins-1)), num_bins+1)
roll_edges = (roll_centers - ((roll_centers[1] - roll_centers[0])/2.)).clone().detach()
rho_edges = (rho_centers - ((rho_centers[1] - rho_centers[0])/2.)).clone().detach()
fov_edges = (fov_centers - ((fov_centers[1] - fov_centers[0])/2.)).clone().detach()
k1_hat_edges = (k1_hat_centers - ((k1_hat_centers[1] - k1_hat_centers[0])/2.)).clone().detach()

pred_roll = (roll_centers[pred['roll'].argmax(1)]).clone().detach()
pred_rho = ((rho_centers[pred['rho'].argmax(1)]) * 0.35).clone().detach()
pred_vfov = (fov_centers[pred['vfov'].argmax(1)]).clone().detach()
pred_k1_hat = (k1_hat_centers[pred['k1_hat'].argmax(1)]).clone().detach()

print(f"Roll (degrees): {pred_roll}, \nDistorted offset of horizon from image centre (unit: ratio of image height before resizing): {pred_rho}, \nvertical field of view (degrees): {pred_vfov}, \nk1_hat (k1/f^2): {pred_k1_hat}")