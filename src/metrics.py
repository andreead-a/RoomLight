"""The three benchmark metrics, all restricted to the pixels of the inserted objects.

    psnr_obj       PSNR of the tonemapped render (gamma 2.2, 8 bit) against the benchmark's
                   tonemapped ground-truth photograph.
    rmse           root mean squared error in linear radiance, against a linear render of
                   the ground-truth scene. Scale-bearing: a wrong exposure costs.
    angular_error  mean per-pixel angle between predicted and true RGB vectors, in degrees.
                   Scale-free: it measures colour, not strength.
"""

import numpy as np


def tonemap(linear):
    """Linear radiance -> gamma-2.2 8-bit image, as the ground-truth jpgs were written."""
    return (np.clip(np.asarray(linear), 0.0, 1.0) ** (1.0 / 2.2) * 255).round().astype(np.uint8)


def masked_psnr(pred, gt, mask):
    """PSNR over the masked pixels of two [0, 1] images."""
    mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def rmse(pred, gt, mask):
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def rgb_angular_error(pred, gt, mask, eps=1e-6):
    """Mean angle between RGB vectors, in degrees, over the masked pixels where both have
    a direction (norm above `eps`)."""
    valid = mask & (np.linalg.norm(pred, axis=-1) > eps) & (np.linalg.norm(gt, axis=-1) > eps)
    a = pred[valid].astype(np.float64)
    b = gt[valid].astype(np.float64)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    # 2 atan2(|a - b|, |a + b|) is exact where arccos(a . b) is ill-conditioned (near 0 deg).
    angle = 2.0 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    return float(np.degrees(angle).mean())
