"""Training losses beyond the per-pixel L1: multi-scale gradients, bright-end peaks, KL."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import circular_pad_2d

REC709_LUMA = (0.2126, 0.7152, 0.0722)


class MultiScaleGradientLoss(nn.Module):
    """L1 between Scharr image gradients over an average-pooled pyramid.

    Supervises edges at the scale they exist, which a plain L1 does not. Panorama
    padding keeps the seam and the poles supervised; average pooling (rather than
    nearest-neighbour subsampling) preserves the energy of small bright sources.
    """

    SCHARR_X = ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0))

    def __init__(self, scales=4):
        super().__init__()
        op_x = torch.tensor(self.SCHARR_X) / 32.0
        self.register_buffer("op_x", op_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("op_y", op_x.t().contiguous().view(1, 1, 3, 3), persistent=False)
        self.scales = scales

    def gradients(self, x):
        channels = x.shape[1]
        padded = circular_pad_2d(x, 1)
        return (F.conv2d(padded, self.op_x.expand(channels, 1, 3, 3), groups=channels),
                F.conv2d(padded, self.op_y.expand(channels, 1, 3, 3), groups=channels))

    def forward(self, predicted, target):
        total, levels = 0.0, 0
        for _ in range(self.scales):
            pred_x, pred_y = self.gradients(predicted)
            target_x, target_y = self.gradients(target)
            total = total + 0.5 * ((pred_x - target_x).abs().mean()
                                   + (pred_y - target_y).abs().mean())
            levels += 1
            if min(predicted.shape[-2:]) < 8:
                break
            predicted = F.avg_pool2d(predicted, 2)
            target = F.avg_pool2d(target, 2)
        return total / levels


def peak_loss(predicted, target, quantile=0.995, eps=1e-6):
    """Log-radiance L1 restricted to the brightest pixels, in linear radiance.

    The mask is the union of the prediction's and the target's top `1 - quantile`
    luminance tail, so a missing light source and a hallucinated one both cost. This is
    the only term that supervises the absolute intensity of the sources, which a log-space
    L1 averages away against the background.

    Args:
        predicted, target: (B, 3, H, W) linear radiance, clamped to a finite range.
    """
    luma = predicted.new_tensor(REC709_LUMA).view(1, 3, 1, 1)
    lum_pred = (predicted * luma).sum(dim=1)
    lum_target = (target * luma).sum(dim=1)
    thr_pred = torch.quantile(lum_pred.flatten(1), quantile, dim=1).view(-1, 1, 1)
    thr_target = torch.quantile(lum_target.flatten(1), quantile, dim=1).view(-1, 1, 1)
    mask = ((lum_pred >= thr_pred) | (lum_target >= thr_target)).unsqueeze(1)
    mask = mask.expand_as(predicted).to(predicted.dtype)
    log_error = ((predicted + eps).log() - (target + eps).log()).abs()
    return (log_error * mask).sum() / mask.sum().clamp_min(1.0)


def gaussian_kl(mean, logvar):
    """KL(q(z|x) || N(0, I)), summed over latent dimensions and averaged over the batch.

    Returns the loss and the detached per-dimension KL (for counting active units).
    """
    kl_per_dim = (0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)).flatten(1).mean(dim=0)
    return kl_per_dim.sum(), kl_per_dim.detach()
