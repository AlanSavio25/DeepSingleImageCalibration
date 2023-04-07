import torch
import numpy as np


def compute_categorical_entropy(x, dim=-1, from_logits=False, eps=1e6):
    r"""Computes normalized categorical entropy along a given tensor dimension. The output is
    dimensionless (i.e. not in nats or bits) because the output is normalized.
    Args:
        x:
            Tensor to compute entropy over
        dim:
            Dimension to compute entropy over. Only relevant for categorical entropy.
        from_logits:
            If ``True``, assume inputs will be unnormalized logits. Otherwise, assume inputs
            are normalized probabilities.
        eps:
            Numerical stabilizer for non-logit inputs
    """
    with torch.no_grad():

        if from_logits:
            p = x.softmax(dim=dim)
            log_p = F.log_softmax(x, dim=dim)
        else:
            p = x
            log_p = x.clamp_min(-eps)

        C = x.shape[dim]
        assert C > 0

        # C-class normalization divisor
        divisor = p.new_tensor(C).log_()

        entropy = log_p.mul(torch.exp(p)).sum(dim=dim).neg().div(divisor)
#         print(entropy)
        assert 0. <= entropy <= 1.
        return entropy


def get_bin_centers(device):
    num_bins = 256

    roll_centers = np.linspace(-45.0, 45.0+(90./(num_bins-1)), num_bins+1)
    roll_edges = torch.tensor(roll_centers - ((roll_centers[1] - roll_centers[0])/2.),
                              dtype=torch.float64,
                              device=device)
    rho_centers = np.linspace(-1., 1.+(2./(num_bins-1)), num_bins+1)
    rho_edges = torch.tensor(rho_centers - ((rho_centers[1] - rho_centers[0])/2.),
                             dtype=torch.float64,
                             device=device)
    fov_centers = np.linspace(20., 105.+(85./(num_bins-1)), num_bins+1)
    fov_edges = torch.tensor(fov_centers - ((fov_centers[1] - fov_centers[0])/2.),
                             dtype=torch.float64, device=device)

    k1_hat_centers = np.linspace(-0.45, 0.+(0.45/(num_bins-1)), num_bins+1)
    k1_hat_edges = torch.tensor(k1_hat_centers - ((k1_hat_centers[1] - k1_hat_centers[0])/2.),
                                dtype=torch.float64, device=device)
    return roll_centers, rho_centers, fov_centers, k1_hat_centers
