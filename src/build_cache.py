#!/usr/bin/env python3
"""Preprocess the train/val EXR directories into one tensor file each.

    python src/build_cache.py /data/panoramas /data/panoramas_cache

`LavalDataset` accepts either form; the cache turns a few thousand EXR reads per training
run into a single `torch.load`.
"""

import sys
from pathlib import Path

import torch

from dataset import LavalDataset

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    out = dst / split
    if out.exists():
        print(f"{split}: cache present, skipping")
        continue
    data = LavalDataset(src / split).data
    torch.save(data, out)
    print(f"{split}: {tuple(data.shape)} -> {out}")
