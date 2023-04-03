import torch
import torchvision
from .base_model import BaseModel
import torch.nn as nn
import numpy as np
from copy import deepcopy
from calib.calib.models.utils_densenet import _DenseBlock, _Transition
from torch.nn import Identity
import pycolmap


class DenseNet(BaseModel):

    default_conf = {
        'name': 'densenet',
        'loss': 'NLL',
        'num_bins': 256,
        'trainable': True,
        'freeze_batch_normalization': False,
        'model': 'densenet161',
        'pretrained': True,  # whether to use ImageNet weights,
        'optimizer': {
            'name': 'basic_optimizer',
        },
    }

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    strict_conf = False

    def _init(self, conf):

        self.conf = conf
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.is_classification = True if self.conf.loss in ['NLL'] else False

        self.num_bins = conf.num_bins

        self.roll_centers = np.linspace(-45.0, 45.0 +
                                        (90./(self.num_bins-1)), self.num_bins+1)
        self.roll_edges = torch.tensor(self.roll_centers - ((self.roll_centers[1] - self.roll_centers[0])/2.),
                                       dtype=torch.float64, device=self.device)

        self.rho_centers = np.linspace(-1., 1. +
                                       (2./(self.num_bins-1)), self.num_bins+1)
        self.rho_edges = torch.tensor(self.rho_centers - ((self.rho_centers[1] - self.rho_centers[0])/2.),
                                      dtype=torch.float64, device=self.device)

        self.fov_centers = np.linspace(
            20., 105.+(85./(self.num_bins-1)), self.num_bins+1)
        self.fov_edges = torch.tensor(self.fov_centers - ((self.fov_centers[1] - self.fov_centers[0])/2.),
                                      dtype=torch.float64, device=self.device)

        self.k1_hat_centers = np.linspace(-0.45, 0. +
                                          (0.45/(self.num_bins-1)), self.num_bins+1)
        self.k1_hat_edges = torch.tensor(self.k1_hat_centers - ((self.k1_hat_centers[1] - self.k1_hat_centers[0])/2.),
                                         dtype=torch.float64, device=self.device)

        Model = getattr(torchvision.models, conf.model)
        self.model = Model(pretrained=self.conf.pretrained)

        layers = []

        # 2208 for 161 layers. 1024 for 121
        num_features = self.model.classifier.in_features
        head_layers = 3
        layers.append(_Transition(num_features, num_features//2))
        num_features = num_features // 2
        growth_rate = 32
        layers.append(_DenseBlock(num_layers=head_layers, num_input_features=num_features,
                                  growth_rate=growth_rate, bn_size=4, drop_rate=0))
        layers.append(nn.BatchNorm2d(num_features+head_layers*growth_rate))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_features+head_layers*growth_rate, 512))
        layers.append(nn.ReLU())
        self.model.classifier = Identity()
        self.model.features.norm5 = Identity()

        if self.is_classification:
            layers.append(nn.Linear(512, self.num_bins))
            layers.append(nn.LogSoftmax(dim=1))
        else:
            layers.append(nn.Linear(512, 1))
            layers.append(nn.Tanh())

        self.roll_head = nn.Sequential(*deepcopy(layers))
        self.rho_head = nn.Sequential(*deepcopy(layers))
        self.vfov_head = nn.Sequential(*deepcopy(layers))
        self.hfov_head = nn.Sequential(*deepcopy(layers))
        self.k1_hat_head = nn.Sequential(*deepcopy(layers))

    def _forward(self, data):
        image = data['image']
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        shared_features = self.model.features(image)
        pred = {}
        if 'roll' in self.conf.heads:
            pred['roll'] = self.roll_head(shared_features)
        if 'rho' in self.conf.heads:
            pred['rho'] = self.rho_head(shared_features)
        if 'vfov' in self.conf.heads:
            pred['vfov'] = self.vfov_head(shared_features)
        if 'hfov' in self.conf.heads:
            pred['hfov'] = self.hfov_head(shared_features)
        if 'k1_hat' in self.conf.heads:
            pred['k1_hat'] = self.k1_hat_head(shared_features)
        return pred

    def loss(self, pred, data):
        loss = {'total': 0}
        if self.conf.loss == 'Huber':
            loss_fn = nn.HuberLoss(reduction='sum')
        elif self.conf.loss == 'L1':
            loss_fn = nn.L1Loss(reduction='sum')
        elif self.conf.loss == 'L2':
            loss_fn = nn.MSELoss(reduction='sum')
        elif self.conf.loss == 'NLL':
            loss_fn = nn.NLLLoss(reduction='sum')

        if 'roll' in self.conf.heads:
            # nbins softmax values if classification, else scalar value
            pred_roll = pred['roll'].squeeze(1)
            if self.is_classification:
                # converted to degrees
                gt_roll = (data['roll'].float()*(180./np.pi))
                gt_roll = torch.bucketize(
                    gt_roll, self.roll_edges) - 1  # converted to class
            else:
                # normalized to [-1,1]
                gt_roll = (data['roll'].float()/(45.*np.pi/180.))
                assert pred_roll.dim() == gt_roll.dim()

            loss_roll = loss_fn(pred_roll, gt_roll)
            loss['roll'] = loss_roll
            loss['total'] += loss_roll

        if 'rho' in self.conf.heads:
            pred_rho = pred['rho'].squeeze(1)
            if self.is_classification:
                gt_rho = (data['rho'].float()/0.35)
                gt_rho = torch.bucketize(gt_rho, self.rho_edges) - 1
            else:
                gt_rho = (data['rho'].float()/0.35)
                assert pred_rho.dim() == gt_rho.dim()

            loss_rho = loss_fn(pred_rho, gt_rho)
            loss['rho'] = loss_rho
            loss['total'] += loss_rho

        if 'vfov' in self.conf.heads:
            pred_vfov = pred['vfov'].squeeze(1)
            if self.is_classification:
                # converted to degrees
                gt_vfov = (data['vfov'].float()*(180./np.pi))
                gt_vfov = torch.bucketize(gt_vfov, self.fov_edges) - 1
            else:
                gt_vfov_deg = (data['vfov'].float()*(180./np.pi))
                min_vfov = 20.
                max_vfov = 105.
                # Normalized [20,105] to [-1,1]
                gt_vfov = (2 * (gt_vfov_deg - min_vfov) /
                           (max_vfov - min_vfov)) - 1
                assert pred_vfov.dim() == gt_vfov.dim()

            loss_vfov = loss_fn(pred_vfov, gt_vfov)
            loss['vfov'] = loss_vfov
            loss['total'] += loss_vfov

        if 'hfov' in self.conf.heads:
            pred_hfov = pred['hfov'].squeeze(1)
            if self.is_classification:
                # converted to degrees
                gt_hfov = (data['hfov'].float()*(180./np.pi))
                gt_hfov = torch.bucketize(gt_hfov, self.fov_edges) - 1
            else:
                gt_hfov_deg = (data['hfov'].float()*(180./np.pi))
                min_hfov = 20.
                max_hfov = 105.
                # Normalized [20,105] to [-1,1]
                gt_hfov = (2 * (gt_hfov_deg - min_hfov) /
                           (max_hfov - min_hfov)) - 1
                assert pred_hfov.dim() == gt_hfov.dim()

            loss_hfov = loss_fn(pred_hfov, gt_hfov)
            loss['hfov'] = loss_hfov
            loss['total'] += loss_hfov

        if 'k1_hat' in self.conf.heads:
            pred_k1_hat = pred['k1_hat'].squeeze(1)
            gt_k1_hat = data['k1_hat'].float()
            if self.is_classification:
                gt_k1_hat = torch.bucketize(gt_k1_hat, self.k1_hat_edges) - 1
            else:
                gt_k1_hat = data['k1_hat'].float()
                min_k1_hat = -0.45
                max_k1_hat = 0.0
                # Normalized [-0.45,0.0] to [-1,1]
                gt_k1_hat = (2 * (gt_k1_hat - min_k1_hat) /
                             (max_k1_hat - min_k1_hat)) - 1
                assert pred_k1_hat.dim() == gt_k1_hat.dim()

            loss_k1_hat = loss_fn(pred_k1_hat, gt_k1_hat)
            loss['k1_hat'] = loss_k1_hat
            loss['total'] += loss_k1_hat
        return loss

    def metrics(self, pred, data):
        metrics = {}
        loss = nn.HuberLoss(reduction='sum')
        l1_loss = nn.L1Loss(reduction='sum')
        l2_loss = nn.MSELoss(reduction='sum')

        # Roll metrics
        if 'roll' in self.conf.heads:
            gt_deg = (data['roll'].float()*(180./np.pi))
            gt_norm = (gt_deg/45.)  # normalized to [-1,1]
            if self.is_classification:
                output = pred['roll']
                pred_class = output.argmax(1)
                pred_deg = torch.tensor(self.roll_centers[pred_class], dtype=torch.float64,
                                        device=self.device)
            else:
                pred_norm = pred['roll'].squeeze(1)
                pred_deg = pred_norm * 45.0
                pred_class = (torch.bucketize(pred_deg, self.roll_edges) - 1)
                assert pred_deg.dim() == gt_deg.dim()
            assert gt_deg.dim() == 1

            pred_deg = pred_deg.to(self.device)
            gt_deg = gt_deg.to(self.device)
            metrics.update({
                'roll/Huber_degree_loss': torch.tensor([loss(pred_deg, gt_deg)]),
                'roll/L1_degree_loss': torch.tensor([l1_loss(pred_deg, gt_deg)]),
                'roll/L2_degree_loss': torch.tensor([l2_loss(pred_deg, gt_deg)]),
            })

        # Rho metrics: pitch degree L1 error, ratio errors
        if 'rho' in self.conf.heads:
            gt_ratio = (data['rho'].float())
            gt_norm = (gt_ratio/0.35).unsqueeze(1)  # normalized to [-1,1]
            if self.is_classification:
                output = pred['rho']
                pred_class = output.argmax(1)
                pred_norm = torch.tensor(self.rho_centers[pred_class], dtype=torch.float64,
                                         device=self.device)
#                 print(f'pred_norm: {pred_norm}')
                pred_norm = pred_norm.unsqueeze(0)
            else:
                pred_norm = pred['rho'].squeeze(1)
                pred_class = (torch.bucketize(pred_norm, self.rho_edges) - 1)

            pred_ratio = pred_norm * 0.35

            # Compute pitch from predicted rho
            H, W = int(data['height'].cpu().item()), int(data['width'].cpu().item())
            F_px = data['f_px'].cpu().item()
            f_ratio = data['focal_length_ratio_height'].cpu().item()
            u0 = H / 2.
            v0 = W / 2.
            k1 = data['k1'].cpu().item()
            k2 = data['k2'].cpu().item()
            predicted_rho = pred_ratio.cpu().item()
            rho_px = predicted_rho * H

            img_pts = [u0, rho_px + v0]
            camera = pycolmap.Camera(
                model='RADIAL',
                width=W,
                height=H,
                params=[F_px, u0, v0, k1, k2],
            )
            normalized_coords = np.array(camera.image_to_world(img_pts))
            camera_no_distortion = pycolmap.Camera(
                model='RADIAL',
                width=W,
                height=H,
                params=[F_px, u0, v0, 0.0, 0.0],
            )
            back_to_image = np.array(
                camera_no_distortion.world_to_image(normalized_coords))
            predicted_tau = (back_to_image[1] - v0) / H

            predicted_pitch = np.arctan(predicted_tau/f_ratio)

            gt_pitch_deg = data['pitch'].cpu().item() * 180/np.pi
            pred_pitch_deg = predicted_pitch * 180/np.pi
            assert gt_ratio.dim() == 1

            pred_ratio = pred_ratio.to(self.device)
            gt_ratio = gt_ratio.to(self.device)
            pred_pitch_deg = torch.tensor(pred_pitch_deg, device=self.device)
            gt_pitch_deg = torch.tensor(gt_pitch_deg, device=self.device)
            metrics.update({
                'rho/Huber_fraction_loss': torch.tensor([loss(pred_ratio, gt_ratio)]),
                'rho/L1_pitch_degree_loss': torch.tensor([l1_loss(pred_pitch_deg, gt_pitch_deg)])
            })

        # V Field of View metrics
        if 'vfov' in self.conf.heads:
            gt_deg = (data['vfov'].float()*(180./np.pi))
            h = data['height'].float()
            gt_pix = 1 / (torch.tan(gt_deg * (np.pi/180.) / 2) * 2 / h)
            if self.is_classification:
                output = pred['vfov']
                pred_class = output.argmax(1)
                pred_deg = torch.tensor(
                    self.fov_centers[pred_class], dtype=torch.float64, device=self.device).unsqueeze(0)
            else:
                pred_norm = pred['vfov'].squeeze(1)
                min_fov = 20.
                max_fov = 105.
                pred_deg = ((pred_norm + 1) *
                            (max_fov - min_fov) / 2) + min_fov
                pred_class = (torch.bucketize(pred_deg, self.fov_edges) - 1)
                assert pred_deg.dim() == gt_deg.dim()

            pred_fy_pix = 1 / (torch.tan(pred_deg * (np.pi/180.) / 2) * 2 / h.to(device=self.device))

            assert gt_deg.dim() == 1
            assert pred_fy_pix.dim() == gt_pix.dim() == 1
            pred_deg = pred_deg.to(self.device)
            gt_deg = gt_deg.to(self.device)
            pred_fy_pix = pred_fy_pix.to(self.device)
            gt_pix = gt_pix.to(self.device)
            metrics.update({
                'vfov/L1_degree_loss': torch.tensor([l1_loss(pred_deg, gt_deg)]),
                'vfov/L1_pixel_loss': torch.tensor([l1_loss(pred_fy_pix, gt_pix)]),
            })

        # Horizontal Field of View metrics
        if 'hfov' in self.conf.heads:
            gt_deg = (data['hfov'].float()*(180./np.pi))
            w = data['width'].float()
            gt_pix = 1 / (torch.tan(gt_deg * (np.pi/180.) / 2) * 2 / w)
            if self.is_classification:
                output = pred['hfov']
                pred_class = output.argmax(1)
                pred_deg = torch.tensor(
                    self.fov_centers[pred_class], dtype=torch.float64, device=self.device).unsqueeze(0)
            else:
                pred_norm = pred['hfov'].squeeze(1)
                min_fov = 20.
                max_fov = 105.
                pred_deg = ((pred_norm + 1) *
                            (max_fov - min_fov) / 2) + min_fov
                pred_class = (torch.bucketize(pred_deg, self.fov_edges) - 1)
                assert pred_deg.dim() == gt_deg.dim()
            pred_fx_px = 1 / (torch.tan(pred_deg * (np.pi/180.) / 2) * 2 / w.to(device=self.device))

            assert gt_deg.dim() == 1
            assert pred_fx_px.dim() == gt_pix.dim() == 1
            pred_deg = pred_deg.to(self.device)
            gt_deg = gt_deg.to(self.device)
            pred_fx_px = pred_fx_px.to(self.device)
            gt_pix = gt_pix.to(self.device)
            metrics.update({
                'hfov/L1_degree_loss': torch.tensor([l1_loss(pred_deg, gt_deg)]),
                'hfov/L1_pixel_loss': torch.tensor([l1_loss(pred_fx_px, gt_pix)]),
            })

        # Average of fx and fy
        if 'hfov' in self.conf.heads and 'vfov' in self.conf.heads:
            metrics.update({
                'avg_fov/L1_pixel_loss': torch.tensor([l1_loss((pred_fx_px + pred_fy_pix)/2, gt_pix)]),
            })

        if 'k1_hat' in self.conf.heads:
            gt_k1_hat = data['k1_hat'].float().to(self.device)
            focal_length_ratio_height = data['focal_length_ratio_height'].float().to(
                self.device)
            gt_k1 = data['k1'].float().to(self.device)
            if self.is_classification:
                output = pred['k1_hat']
                pred_class = output.argmax(1)
                pred_k1_hat = torch.tensor(
                    self.k1_hat_centers[pred_class], dtype=torch.float64, device=self.device).unsqueeze(0)
            else:
                pred_norm_k1_hat = pred['k1_hat'].squeeze(1)
                min_k1_hat = -0.45
                max_k1_hat = 0.0
                pred_k1_hat = ((pred_norm_k1_hat + 1) *
                               (max_k1_hat - min_k1_hat) / 2) + min_k1_hat
                pred_class = (torch.bucketize(
                    pred_k1_hat, self.k1_hat_edges) - 1)
                assert pred_k1_hat.dim() == gt_k1_hat.dim()

            pred_k1_hat = pred_k1_hat.to(self.device)
            gt_k1_hat = gt_k1_hat.to(self.device)
            pred_k1 = pred_k1_hat * focal_length_ratio_height**2
            metrics.update({
                'k1_hat/L1_loss': torch.tensor([l1_loss(pred_k1_hat, gt_k1_hat)]),
                'k1_hat/L1_k1_loss': torch.tensor([l1_loss(pred_k1, gt_k1)]),
            })

            assert gt_k1_hat.dim() == 1

        return metrics
