"""Panorama dataset: 4-channel EXRs (HDR radiance + metric depth) at 128 x 256.

Each EXR holds linear HDR radiance in channels 0-2 and metric depth in channel 3. The
network sees log1p(radiance) and log(depth), z-scored per channel with the training-set
statistics below; `denormalize` inverts that.

`root_dir` is either a directory of EXRs or a tensor file written by `build_cache.py`.
"""

import random
from pathlib import Path

import numpy as np
import pyexr
import torch
import torchvision.transforms as TVT
from torch.utils.data import Dataset
from tqdm import tqdm

from pano import equirect_tilt


class LavalDataset(Dataset):
    LOG_MEAN = (0.9586, 0.7788, 0.4920, 0.0020)
    LOG_STD = (0.6250, 0.5933, 0.5579, 0.4601)
    normalize = TVT.Normalize(mean=LOG_MEAN, std=LOG_STD)

    def __init__(self, root_dir, train=False, tilt_deg=0.0):
        """
        Args:
            root_dir: directory of .exr panoramas, or a cached tensor file.
            train: enable the augmentations (random roll, mirror, tilt).
            tilt_deg: maximum camera tilt in degrees, drawn uniformly per sample.
        """
        self.train = train
        self.tilt_deg = float(tilt_deg)

        root = Path(root_dir)
        if root.is_dir():
            images = []
            for path in tqdm(sorted(root.glob("*.exr")), desc=f"loading {root.name}"):
                image = torch.from_numpy(pyexr.read(str(path))).permute(2, 0, 1)
                image_log = torch.cat([torch.log1p(image[:3]), torch.log(image[3:])], dim=0)
                images.append(self.normalize(image_log))
            if not images:
                raise RuntimeError(f"no EXR images found in {root}")
            self.data = torch.stack(images)
        else:
            self.data = torch.load(root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.train:
            image = torch.roll(image, shifts=random.randint(0, image.shape[-1] - 1), dims=2)
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
            # After the roll, so the fixed tilt axis is uniformly random in azimuth.
            if self.tilt_deg > 0.0:
                image = equirect_tilt(image, np.deg2rad(random.uniform(-self.tilt_deg,
                                                                       self.tilt_deg)))
        return image

    @staticmethod
    def denormalize(image_norm):
        """(B, 4, H, W) or (4, H, W) normalised -> (B, 4, H, W) linear radiance + depth."""
        if image_norm.dim() == 3:
            image_norm = image_norm.unsqueeze(0)
        std = image_norm.new_tensor(LavalDataset.LOG_STD).view(-1, 1, 1)
        mean = image_norm.new_tensor(LavalDataset.LOG_MEAN).view(-1, 1, 1)
        image_log = image_norm * std + mean
        return torch.cat([torch.expm1(image_log[:, :3]), torch.exp(image_log[:, 3:])], dim=1)
