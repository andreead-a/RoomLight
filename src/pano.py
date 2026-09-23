"""Equirectangular geometry: pixel directions, camera tilt, and the emitting shell mesh.

Conventions (shared by every module): y is up, row i has polar angle
theta = (i + 0.5) * pi / H, column j has azimuth phi = (j + 0.5) * 2 pi / W, and the
direction of a pixel is (sin(phi) sin(theta), cos(theta), -cos(phi) sin(theta)).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from models import circular_pad_2d


def equirect_directions(height, width, device=None, dtype=torch.float32):
    """Unit direction of every pixel centre, as a (3, H, W) tensor."""
    theta = ((torch.arange(height, device=device, dtype=dtype) + 0.5)
             * (math.pi / height)).view(height, 1)
    phi = ((torch.arange(width, device=device, dtype=dtype) + 0.5)
           * (2.0 * math.pi / width)).view(1, width)
    sin_t = theta.sin()
    return torch.stack([phi.sin() * sin_t,
                        theta.cos().expand(height, width),
                        -phi.cos() * sin_t])


def equirect_tilt(image, angle_rad):
    """Rotate an equirectangular map about the +x axis by `angle_rad` (a camera tilt).

    Bilinear resampling on top of panorama padding, so the seam and the poles resample
    continuously. Depth is a radial distance and needs no correction.

    Args:
        image: (C, H, W) or (B, C, H, W).
    """
    single = image.dim() == 3
    if single:
        image = image.unsqueeze(0)
    height, width = image.shape[-2:]

    # Rotate the output directions backwards to find where to sample the input.
    x, y, z = equirect_directions(height, width, image.device, image.dtype)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    y_src = cos_a * y + sin_a * z
    z_src = cos_a * z - sin_a * y

    # atan2 rather than acos for theta: acos is ill-conditioned near the poles.
    v = torch.atan2(torch.sqrt(x * x + z_src * z_src), y_src) * (1.0 / math.pi)
    u = torch.atan2(x, -z_src) * (0.5 / math.pi) % 1.0

    pad = 1
    padded = circular_pad_2d(image, pad)
    grid_x = 2.0 * (u * width + pad) / (width + 2 * pad) - 1.0
    grid_y = 2.0 * (v * height + pad) / (height + 2 * pad) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)[None].expand(image.shape[0], -1, -1, -1)
    out = F.grid_sample(padded, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out[0] if single else out


def shell_topology(height, width):
    """Faces and texture coordinates of the emitting shell built from an H x W depth map.

    The vertices are, in order: one per pixel (H*W), a duplicate of column 0 to close the
    longitude seam (H), and W duplicates each of the north and south pole (2*W), so the
    texture does not pinch at the poles. `shell_vertices` produces positions in the same
    order. Texture coordinates follow Mitsuba's convention (v = 0 at the top row).

    Returns:
        faces: (F, 3) int64 array.
        uv: (V, 2) float32 array.
    """
    H, W = height, width
    seam_base = H * W
    north_base = seam_base + H
    south_base = north_base + W

    ys = np.arange(H - 1)[:, None]
    xs = np.arange(W)[None, :]
    i0 = ys * W + xs
    i2 = (ys + 1) * W + xs
    i1 = np.where(xs < W - 1, ys * W + xs + 1, seam_base + ys)
    i3 = np.where(xs < W - 1, (ys + 1) * W + xs + 1, seam_base + ys + 1)
    faces = np.stack([np.stack([i0, i2, i1], axis=-1),
                      np.stack([i1, i2, i3], axis=-1)], axis=-2).reshape(-1, 3)

    cols = np.arange(W)
    top_next = np.where(cols < W - 1, cols + 1, seam_base)
    north_faces = np.stack([north_base + cols, cols, top_next], axis=-1)
    bottom = (H - 1) * W
    bottom_next = np.where(cols < W - 1, bottom + cols + 1, seam_base + (H - 1))
    south_faces = np.stack([south_base + cols, bottom_next, bottom + cols], axis=-1)
    faces = np.concatenate([faces, north_faces, south_faces], axis=0)

    u = np.linspace(0.5 / W, 1.0 - 0.5 / W, W)
    v = np.linspace(0.5 / H, 1.0 - 0.5 / H, H)
    uu, vv = np.meshgrid(u, v)
    uv = np.concatenate([
        np.stack([uu, vv], axis=-1).reshape(-1, 2),
        np.stack([np.full(H, 1.0 + 0.5 / W), v], axis=-1),
        np.stack([u, np.zeros(W)], axis=-1),
        np.stack([u, np.ones(W)], axis=-1),
    ], axis=0)
    return faces.astype(np.int64), uv.astype(np.float32)


def shell_vertices(depth):
    """Vertex positions of the shell for an (H, W) depth map, in `shell_topology` order.

    Differentiable in `depth`. Returns an (H*W + H + 2*W, 3) tensor.
    """
    H, W = depth.shape
    dirs = equirect_directions(H, W, depth.device, depth.dtype).permute(1, 2, 0)
    vertices = (dirs * depth.unsqueeze(-1)).reshape(-1, 3)
    seam = vertices[torch.arange(H, device=depth.device) * W]
    north = vertices[:W].mean(dim=0, keepdim=True).repeat(W, 1)
    south = vertices[(H - 1) * W:].mean(dim=0, keepdim=True).repeat(W, 1)
    return torch.cat([vertices, seam, north, south], dim=0)
