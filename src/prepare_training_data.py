#!/usr/bin/env python3
"""Build the training panoramas from the Laval Indoor HDR dataset.

    python src/prepare_training_data.py data/IndoorHDRDataset2018 data/panoramas/train

Every input EXR (an HDR equirectangular panorama with a black hole at the nadir, where the
tripod was) becomes a 128 x 256 four-channel EXR: linear RGB radiance, exposure-normalised
so its log-average luminance is 1, and aligned depth. Four steps:

    1. resize to 1024 x 2048 and inpaint the nadir hole with LaMa, working on a gnomonic
       view of the nadir in a tonemapped domain and mapping the result back to HDR;
    2. estimate depth with MoGe-2 on the tonemapped panorama: 12 perspective views on an
       icosahedron are predicted separately and merged into one 512 x 1024 distance map
       by solving for the log distance whose gradients and Laplacian match the views, with
       the top and bottom rows tied together so the poles close;
    3. smooth the depth around both poles, where the merge leaves a seam;
    4. normalise the exposure and downsample both to 128 x 256 with area averaging under
       panorama padding.

Extra dependencies: `simple-lama-inpainting`, MoGe
(`pip install git+https://github.com/microsoft/MoGe.git`) and its `utils3d`. The models
are downloaded on first use. Already-written outputs are skipped, so the script resumes.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyexr
import torch
from scipy.sparse import lil_matrix, vstack
from scipy.sparse.linalg import lsmr
from tqdm import tqdm

PANO_SIZE = (1024, 2048)     # (H, W) the raw panoramas are resized to before inpainting
DEPTH_SIZE = (512, 1024)     # (H, W) of the merged depth map
OUT_SIZE = (128, 256)        # (H, W) of the training panoramas
INPAINT_FOV, INPAINT_SIZE = 80.0, 512   # gnomonic nadir view the hole is inpainted in
BLUR_FOV, BLUR_SIZE = 15.0, 128         # gnomonic polar views the depth is smoothed in
REC709_LUMA = (0.2126, 0.7152, 0.0722)


# ---------------------------------------------------------------------------
# Equirectangular <-> gnomonic nadir view
# ---------------------------------------------------------------------------

def _bilinear(image, x, y):
    """Sample an (H, W, C) image at fractional pixel coordinates, clamped at the border."""
    h, w = image.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    image = image.astype(np.float32)
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x1] * fx * (1 - fy)
            + image[y1, x0] * (1 - fx) * fy + image[y1, x1] * fx * fy)


def equirect_to_nadir(equirect, size, fov_deg):
    """Gnomonic view of the nadir (camera looking straight down), size x size pixels."""
    h, w = equirect.shape[:2]
    tan_half = np.tan(np.radians(fov_deg / 2.0))
    gx, gy = np.meshgrid(np.linspace(-tan_half, tan_half, size), np.linspace(-tan_half, tan_half, size))
    norm = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    rx, ry, rz = gx / norm, -1.0 / norm, gy / norm
    lon, lat = np.arctan2(rx, rz), np.arcsin(np.clip(ry, -1.0, 1.0))
    return _bilinear(equirect, (lon / (2 * np.pi) + 0.5) * w, (0.5 - lat / np.pi) * h)


def nadir_to_equirect(nadir, size, fov_deg):
    """Inverse of `equirect_to_nadir`: an (H, W, C) map, zero outside the nadir view."""
    h, w = size
    n = nadir.shape[0]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    lon, lat = (u / w - 0.5) * (2 * np.pi), (0.5 - v / h) * np.pi
    rx, ry, rz = np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)
    down = ry < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = np.where(down, rx / -ry, 0.0)
        gz = np.where(down, rz / -ry, 0.0)
    tan_half = np.tan(np.radians(fov_deg / 2.0))
    inside = down & (np.abs(gx) <= tan_half) & (np.abs(gz) <= tan_half)
    sampled = _bilinear(nadir, (gx / tan_half + 1.0) * 0.5 * (n - 1),
                        (gz / tan_half + 1.0) * 0.5 * (n - 1))
    out = np.zeros((h, w, nadir.shape[-1]), dtype=np.float32)
    out[inside] = sampled[inside]
    return out


# ---------------------------------------------------------------------------
# Step 1: nadir inpainting
# ---------------------------------------------------------------------------

def luminance(image):
    return image[..., 0] * REC709_LUMA[0] + image[..., 1] * REC709_LUMA[1] + image[..., 2] * REC709_LUMA[2]


def reinhard_constants(hdr, key=0.18, eps=1e-6):
    """Exposure and white point of the extended Reinhard operator, from the image itself."""
    lum = luminance(hdr)
    scale = key / np.exp(np.mean(np.log(lum + eps)))
    return {"scale": scale, "white": np.percentile(lum * scale, 99.5)}


def reinhard_tonemap(hdr, constants, eps=1e-6):
    """Linear HDR -> gamma-2.2 LDR in [0, 1] with fixed constants, so patches are consistent."""
    lum = luminance(hdr) * constants["scale"]
    lum_tm = lum * (1 + lum / constants["white"] ** 2) / (1 + lum)
    ldr = hdr * constants["scale"] * (lum_tm / (lum + eps))[..., None]
    return np.clip(np.power(ldr, 1 / 2.2), 0, 1)


def reinhard_inverse(ldr, constants, eps=1e-6):
    """The inverse of `reinhard_tonemap`, solving the quadratic in the luminance."""
    linear = np.power(np.clip(ldr, eps, 1), 2.2)
    lum_tm = luminance(linear)
    a, b, c = (1 - lum_tm) / constants["white"] ** 2, 1 - lum_tm, -lum_tm
    lum = (-b + np.sqrt(np.clip(b ** 2 - 4 * a * c, 0, None))) / (2 * a + eps)
    return linear * (lum / (lum_tm + eps))[..., None] / constants["scale"]


def inpaint_nadir(hdr, lama, device):
    """Fill the black nadir hole. Returns (HDR panorama, tonemapped uint8 panorama)."""
    constants = reinhard_constants(hdr)
    nadir = equirect_to_nadir(hdr, INPAINT_SIZE, INPAINT_FOV)
    hole = np.all(nadir == 0.0, axis=-1).astype(np.uint8)
    hole = cv2.dilate(hole, np.ones((7, 7), np.uint8)) > 0

    image = torch.from_numpy(reinhard_tonemap(nadir, constants)).permute(2, 0, 1)[None].float()
    mask = torch.from_numpy(hole.astype(np.float32))[None, None]
    with torch.inference_mode():
        filled = lama.model(image.to(device), mask.to(device))[0].permute(1, 2, 0).cpu().numpy()

    filled_equirect = nadir_to_equirect(filled, PANO_SIZE, INPAINT_FOV)
    hole_equirect = nadir_to_equirect(hole[..., None].astype(np.float32), PANO_SIZE, INPAINT_FOV)
    hdr = hdr * (1 - hole_equirect) + reinhard_inverse(filled_equirect, constants) * hole_equirect
    ldr = reinhard_tonemap(hdr, constants)
    return hdr.astype(np.float32), np.uint8(np.round(ldr * 255).clip(0, 255))


# ---------------------------------------------------------------------------
# Step 2: depth from MoGe-2, merged over 12 perspective views
# ---------------------------------------------------------------------------

def merge_panorama_depth(width, height, distance_maps, pred_masks, extrinsics, intrinsics,
                         cross_pole_weight=10.0, adjacent_weight=10.0):
    """MoGe's panorama merge, plus constraints that close the two poles.

    The per-view log distances are combined by least squares on their gradients and
    Laplacians (as in `moge.utils.panorama.merge_panorama_depth`). Two extra sets of
    equations tie the top and bottom rows: neighbouring pixels of each polar row should
    agree (`adjacent_weight`) and so should pixels a half turn apart (`cross_pole_weight`),
    since every pixel of a polar row looks at nearly the same point.
    """
    import utils3d
    from scipy.ndimage import convolve
    from moge.utils.panorama import (grad_equation, poisson_equation,
                                     spherical_uv_to_directions)

    if max(width, height) > 256:
        init, _ = merge_panorama_depth(width // 2, height // 2, distance_maps, pred_masks,
                                       extrinsics, intrinsics, cross_pole_weight, adjacent_weight)
        init = cv2.resize(init, (width, height), cv2.INTER_LINEAR)
    else:
        init = None

    directions = spherical_uv_to_directions(utils3d.np.uv_map(height, width))
    grads_x, grads_y, masks_x, masks_y, laplacians, laplacian_masks, view_masks = ([] for _ in range(7))
    for i in range(len(distance_maps)):
        uv, depth = utils3d.np.project_cv(directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        valid = (depth > 0) & (uv > 0).all(axis=-1) & (uv < 1).all(axis=-1)
        pixels = utils3d.np.uv_to_pixel(np.clip(uv, 0, 1), distance_maps[i].shape).astype(np.float32)
        log_distance = np.where(valid, cv2.remap(np.log(distance_maps[i]), pixels[..., 0], pixels[..., 1],
                                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0)
        mask = valid & (cv2.remap(pred_masks[i].astype(np.uint8), pixels[..., 0], pixels[..., 1],
                                  cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0)

        padded = np.pad(log_distance, ((0, 0), (0, 1)), mode="wrap")
        grads_x.append(padded[:, :-1] - padded[:, 1:])
        grads_y.append(padded[:-1, :] - padded[1:, :])
        padded = np.pad(mask, ((0, 0), (0, 1)), mode="wrap")
        masks_x.append(padded[:, :-1] & padded[:, 1:])
        masks_y.append(padded[:-1, :] & padded[1:, :])

        padded = np.pad(np.pad(log_distance, ((1, 1), (0, 0)), mode="edge"), ((0, 0), (1, 1)), mode="wrap")
        laplacians.append(convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32))[1:-1, 1:-1])
        padded = np.pad(np.pad(mask, ((1, 1), (0, 0)), mode="edge"), ((0, 0), (1, 1)), mode="wrap")
        laplacian_masks.append(convolve(padded.astype(np.uint8),
                                        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))[1:-1, 1:-1] == 5)
        view_masks.append(mask)

    def average(values, masks):
        values, masks = np.stack(values), np.stack(masks)
        return np.sum(values * masks, axis=0) / np.sum(masks, axis=0).clip(1e-3), np.any(masks, axis=0).reshape(-1)

    grad_x, mask_x = average(grads_x, masks_x)
    grad_y, mask_y = average(grads_y, masks_y)
    laplacian, laplacian_mask = average(laplacians, laplacian_masks)

    # Pole constraints: adjacent and diametrically opposite pixels of the top and bottom rows.
    half = width // 2
    poles = lil_matrix((3 * width, width * height))
    bottom = (height - 1) * width
    row = 0
    for i in range(width):
        for offset in (0, bottom):
            poles[row, offset + i] = adjacent_weight
            poles[row, offset + (i + 1) % width] = -adjacent_weight
            row += 1
    for i in range(half):
        for offset in (0, bottom):
            poles[row, offset + i] = cross_pole_weight
            poles[row, offset + i + half] = -cross_pole_weight
            row += 1

    A = vstack([grad_equation(width, height, wrap_x=True)[np.concatenate([mask_x, mask_y])],
                poisson_equation(width, height, wrap_x=True)[laplacian_mask],
                poles.tocsr()])
    b = np.concatenate([grad_x.reshape(-1)[mask_x], grad_y.reshape(-1)[mask_y],
                        laplacian.reshape(-1)[laplacian_mask], np.zeros(3 * width)])
    x, *_ = lsmr(A, b, atol=1e-5, btol=1e-5,
                 x0=np.log(init).reshape(-1) if init is not None else None)
    return np.exp(x).reshape(height, width).astype(np.float32), np.any(view_masks, axis=0)


class PanoramaDepth:
    """MoGe-2 depth for an equirectangular LDR panorama, in metres."""

    def __init__(self, device, view_resolution=512, batch_size=4):
        import utils3d
        from moge.model.v2 import MoGeModel
        from moge.utils.panorama import get_panorama_cameras
        self.device = device
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device).eval()
        self.extrinsics, self.intrinsics = get_panorama_cameras()
        self.resolution = view_resolution
        self.batch_size = batch_size
        fov_x, _ = np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(self.intrinsics)))
        self.fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)

    def __call__(self, ldr):
        from moge.utils.panorama import split_panorama_image
        views = split_panorama_image(ldr, self.extrinsics, self.intrinsics, self.resolution)
        images = torch.tensor(np.stack(views) / 255, dtype=torch.float32,
                              device=self.device).permute(0, 3, 1, 2)
        distance, mask = [], []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                stop = start + self.batch_size
                output = self.model.infer(images[start:stop], fov_x=self.fov_x[start:stop],
                                          apply_mask=False)
                distance.append(output["points"].norm(dim=-1).cpu().numpy())
                mask.append(output["mask"].cpu().numpy())
        depth, _ = merge_panorama_depth(DEPTH_SIZE[1], DEPTH_SIZE[0], np.concatenate(distance),
                                        np.concatenate(mask), self.extrinsics, self.intrinsics)
        return depth


# ---------------------------------------------------------------------------
# Steps 3 and 4: polar smoothing, exposure normalisation, downsampling
# ---------------------------------------------------------------------------

def blur_pole(depth, fov_deg, size, sigma=15.0):
    """Gaussian-blur the depth in a gnomonic view of the nadir, fading out towards its edge."""
    nadir = equirect_to_nadir(depth, size, fov_deg)[..., 0]
    y, x = np.indices(nadir.shape)
    radius = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    weight = np.exp(-(radius / radius.max() / 0.4) ** 2).astype(np.float32)
    blurred = cv2.GaussianBlur(nadir, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smoothed = nadir_to_equirect((nadir * (1 - weight) + blurred * weight)[..., None],
                                 depth.shape[:2], fov_deg)
    return np.where(smoothed > 0, smoothed, depth)


def smooth_poles(depth):
    """Smooth an (H, W, 1) depth map around the nadir and (by flipping) the zenith."""
    depth = blur_pole(depth, BLUR_FOV, BLUR_SIZE)
    return blur_pole(depth[::-1], BLUR_FOV, BLUR_SIZE)[::-1]


def circular_pad(image, pad):
    """Panorama padding of an (H, W, C) map: wrap in longitude, flip-and-roll across the poles."""
    w = image.shape[1]
    top = np.flip(np.roll(image[:pad], w // 2, axis=1), axis=0)
    bottom = np.flip(np.roll(image[-pad:], w // 2, axis=1), axis=0)
    image = np.concatenate([top, image, bottom], axis=0)
    return np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)


def downsample(image, size):
    """Area-averaging downsample of an (H, W, C) panorama to (H', W'), padded so the seams
    and the poles average over their true neighbours."""
    h, w = image.shape[:2]
    factor = h / size[0]
    pad = int(np.ceil(factor))
    padded = circular_pad(image, pad)
    resized = cv2.resize(padded, (round(padded.shape[1] / factor), round(padded.shape[0] / factor)),
                         interpolation=cv2.INTER_AREA)
    crop = round(pad / factor)
    return resized[crop:crop + size[0], crop:crop + size[1]]


def assemble(hdr, depth):
    """Inpainted 1024 x 2048 HDR + 512 x 1024 depth -> 128 x 256 x 4 training panorama."""
    hdr = downsample(hdr, DEPTH_SIZE)
    hdr = hdr / np.exp(np.mean(np.log(luminance(hdr) + 1e-6)))   # log-average luminance -> 1
    depth = smooth_poles(depth[..., None] if depth.ndim == 2 else depth)
    return downsample(np.concatenate([hdr, depth], axis=-1), OUT_SIZE).astype(np.float32)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("source_dir", type=Path, help="directory of the raw Laval EXR panoramas")
    parser.add_argument("output_dir", type=Path, help="where the 128 x 256 EXRs are written")
    parser.add_argument("--shard", default="0/1", metavar="INDEX/COUNT",
                        help="process every COUNT-th panorama starting at INDEX, to split "
                             "the work over several jobs")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    shard_index, shard_count = (int(v) for v in args.shard.split("/"))

    from simple_lama_inpainting import SimpleLama
    device = torch.device(f"cuda:{args.gpu}")
    lama = SimpleLama(device)
    panorama_depth = PanoramaDepth(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(sorted(args.source_dir.glob("*.exr"))[shard_index::shard_count]):
        out = args.output_dir / path.name
        if out.exists():
            continue
        hdr = downsample(pyexr.read(str(path))[..., :3], PANO_SIZE)
        hdr, ldr = inpaint_nadir(hdr, lama, device)
        depth = panorama_depth(ldr)
        pyexr.write(str(out), assemble(hdr, depth))


if __name__ == "__main__":
    main()
