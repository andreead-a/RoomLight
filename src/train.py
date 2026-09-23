#!/usr/bin/env python3
"""Train the panorama VAE.

    python src/train.py --config configs/train.yaml
    python src/train.py --config configs/train.yaml output_dir=outputs/my_run train.lr=1e-4

Any config key can be overridden on the command line in dotlist form. The resolved config
is written to `<output_dir>/config.yaml`, which is what `models.load_model` reads later.
Losses and image panels are logged to wandb (set WANDB_MODE=offline to log locally).
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import LavalDataset
from losses import MultiScaleGradientLoss, gaussian_kl, peak_loss
from models import ConvVAE

RADIANCE_MAX = 1e6      # expm1 of an early decoder output can overflow; keep it finite
DISPLAY_EXPOSURE = 0.18  # exposure of the RGB panels logged to wandb


def linear_radiance(normalized):
    return LavalDataset.denormalize(normalized)[:, :3].clamp(0.0, RADIANCE_MAX)


def to_uint8(image):
    return np.uint8(np.round(np.clip(image, 0.0, 1.0) ** (1.0 / 2.2) * 255))


def image_panels(prefix, images, n=10):
    """wandb panels of a denormalised (B, 4, H, W) batch: RGB, log radiance, log depth."""
    grid = vutils.make_grid(images[:n], nrow=5, padding=2).detach().cpu().permute(1, 2, 0).numpy()
    log_rgb = np.log(grid[..., :3].mean(-1) + 1e-6)
    log_depth = np.log(grid[..., 3] + 1e-6)
    return {
        f"{prefix}/rgb": wandb.Image(to_uint8(grid[..., :3] * DISPLAY_EXPOSURE)),
        f"{prefix}/log_radiance": wandb.Image(plt.get_cmap("inferno")((log_rgb + 4.0) / 10.0)),
        f"{prefix}/log_depth": wandb.Image(plt.get_cmap("viridis")((log_depth + 1.2) / 3.3)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the panorama VAE.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(overrides))
    train_cfg, loss_cfg = cfg.train, cfg.train.loss
    device = torch.device(f"cuda:{args.gpu}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))

    model = ConvVAE(**OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    print(f"ConvVAE: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    data_root = Path(cfg.data_root)
    train_set = LavalDataset(data_root / "train", train=True, tilt_deg=train_cfg.tilt_deg)
    val_set = LavalDataset(data_root / "val", train=False)
    loader_kwargs = dict(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
                         persistent_workers=train_cfg.num_workers > 0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    # AdamW, linear warmup then cosine decay to `min_lr`.
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  betas=tuple(train_cfg.betas),
                                  weight_decay=train_cfg.weight_decay)
    total_iters = train_cfg.epochs * len(train_loader)
    warmup = train_cfg.warmup_iters
    min_lr_frac = train_cfg.min_lr / train_cfg.lr

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(total_iters - warmup, 1), 1.0)
        return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    print(f"{train_cfg.epochs} epochs x {len(train_loader)} iterations, "
          f"KL annealed over the first {loss_cfg.kl_anneal_iters}")

    grad_loss = MultiScaleGradientLoss(scales=loss_cfg.grad_scales).to(device)
    weights = {"hdr": 1.0, "depth": 1.0, "grad_hdr": loss_cfg.w_grad_hdr,
               "grad_depth": loss_cfg.w_grad_depth, "peak": loss_cfg.w_peak}

    def compute_losses(x, kl_weight, sample_posterior):
        decoded, posterior = model(x, sample_posterior=sample_posterior)
        terms = {"hdr": (decoded[:, :3] - x[:, :3]).abs().mean(),
                 "depth": (decoded[:, 3:] - x[:, 3:]).abs().mean()}
        if weights["grad_hdr"] > 0:
            terms["grad_hdr"] = grad_loss(decoded[:, :3], x[:, :3])
        if weights["grad_depth"] > 0:
            terms["grad_depth"] = grad_loss(decoded[:, 3:], x[:, 3:])
        if weights["peak"] > 0:
            terms["peak"] = peak_loss(linear_radiance(decoded), linear_radiance(x),
                                      quantile=loss_cfg.peak_quantile)
        kl, kl_per_dim = gaussian_kl(posterior.mean, posterior.logvar)
        loss = kl_weight * kl + sum(weights[k] * v for k, v in terms.items())
        terms.update(kl=kl, loss=loss)
        return decoded, terms, kl_per_dim

    run = wandb.init(project=cfg.wandb_project, name=out_dir.name,
                     config=OmegaConf.to_container(cfg, resolve=True))
    run.define_metric("*", step_metric="step")
    z_fixed = torch.randn(10, model.latent_dim, device=device,
                          generator=torch.Generator(device=device).manual_seed(1234))

    step = 0
    for epoch in range(train_cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        for batch in train_loader:
            step += 1
            x = batch.to(device, non_blocking=True)
            kl_weight = min(step / loss_cfg.kl_anneal_iters, 1.0) * loss_cfg.w_kl
            optimizer.zero_grad()
            decoded, terms, kl_per_dim = compute_losses(x, kl_weight, sample_posterior=True)
            terms["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            log = {f"train/{k}": v.item() for k, v in terms.items()}
            log.update({"step": step, "lr": scheduler.get_last_lr()[0], "kl_weight": kl_weight,
                        "train/grad_norm": grad_norm.item(),
                        "train/active_units": int((kl_per_dim > 0.01).sum())})
            if step % 50 == 0:
                log.update(image_panels("train/input", LavalDataset.denormalize(x)))
                log.update(image_panels("train/output", LavalDataset.denormalize(decoded)))
            wandb.log(log)
        train_secs = time.perf_counter() - t0

        last_epoch = epoch == train_cfg.epochs - 1
        if (epoch + 1) % train_cfg.val_every == 0 or last_epoch:
            model.eval()
            sums, count = {}, 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    decoded, terms, _ = compute_losses(x, kl_weight, sample_posterior=False)
                    for k, v in terms.items():
                        sums[k] = sums.get(k, 0.0) + v.item() * len(x)
                    count += len(x)
                log = {f"val/{k}": v / count for k, v in sums.items()}
                log["step"] = step
                log.update(image_panels("val/input", LavalDataset.denormalize(x)))
                log.update(image_panels("val/output", LavalDataset.denormalize(decoded)))
                log.update(image_panels("val/samples_fixed",
                                        LavalDataset.denormalize(model.decode(z_fixed))))
                log.update(image_panels("val/samples", LavalDataset.denormalize(
                    model.decode(torch.randn_like(z_fixed)))))
            wandb.log(log)
            torch.save({"epoch": epoch, "step": step, "state_dict": model.state_dict()},
                       out_dir / "last_epoch.pth")
        if train_cfg.ckpt_every and (epoch + 1) % train_cfg.ckpt_every == 0:
            torch.save({"epoch": epoch, "step": step, "state_dict": model.state_dict()},
                       out_dir / f"epoch_{epoch + 1}.pth")
        print(f"epoch {epoch + 1}/{train_cfg.epochs}: {train_secs:.0f}s, "
              f"loss {terms['loss'].item():.4f}", flush=True)


if __name__ == "__main__":
    main()
