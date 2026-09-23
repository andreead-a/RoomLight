#!/usr/bin/env python3
"""Inverse rendering on the synthetic benchmark: fit an environment to one photograph.

For every (scene, material) the environment is optimised so that a differentiable Mitsuba
render of the `input` view (two inserted objects) matches the photograph. Three modes:

    prior_al       a 512-d latent and a scalar log-exposure through the frozen decoder;
                   the decoded panorama with depth are lifted into a 2.5D area light.
    direct_al      no decoder: free log-radiance and log-depth per pixel which define a
                   2.5D area light, with a smoothness prior on both fields.
    direct_envmap  free log-radiance per pixel on an environment map at infinity.

    python src/fit.py prior_al outputs/checkpoint
    python src/fit.py direct_al
    python src/fit.py direct_envmap --scenes bathroom --materials mirror -o optim.spp=128

Each case writes `best_env.exr` (RGB radiance + depth at the best iterate), `best.pth`,
`history.json` and `loss.png` to `<save_dir>/<run name>/<scene>/<material>/`; the run's
resolved config goes to `<save_dir>/<run name>/config.yaml`. `evaluate.py` scores them.
"""

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyexr
import torch
import trimesh
from omegaconf import OmegaConf
from tqdm import tqdm

import render
from dataset import LavalDataset
from metrics import tonemap
from models import load_model
from pano import shell_vertices

RADIANCE_MAX = 1e6
MODES = {"prior_al": "mesh", "direct_al": "mesh", "direct_envmap": "envmap"}


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    scene: str
    material: str
    scene_dir: Path
    target: torch.Tensor   # [H, W, 3] linear radiance of the photograph
    mask: torch.Tensor     # [H, W, 1], 1 on the inserted objects
    points: torch.Tensor   # [N, 3] object surface points, for the repulsion term


def list_scenes(cfg):
    if cfg.scene_list:
        names = [line.strip() for line in Path(cfg.scene_list).read_text().splitlines()
                 if line.strip()]
    else:
        names = sorted(d.name for d in Path(cfg.source_dir).iterdir() if d.is_dir())
    return names


def load_points(path, voxel):
    """Object point cloud thinned to one point (the centroid) per `voxel` cube."""
    points = np.asarray(trimesh.load(str(path)).vertices, dtype=np.float64)
    _, inverse, counts = np.unique(np.floor(points / voxel).astype(np.int64), axis=0,
                                   return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse.ravel(), points)
    return sums / counts[:, None]


def load_case(source_dir, scene, material, optim, device):
    scene_dir = Path(source_dir) / scene
    photo = cv2.imread(str(scene_dir / f"{material}_input_rendered.jpg"))[:, :, ::-1]
    mask = cv2.imread(str(scene_dir / "input_mask.png"))[:, :, :1]
    return Case(
        scene=scene, material=material, scene_dir=scene_dir,
        target=torch.tensor((photo / 255.0) ** 2.2, dtype=torch.float32, device=device),
        mask=torch.tensor(mask / 255.0, dtype=torch.float32, device=device),
        points=torch.tensor(load_points(scene_dir / "objects_input.ply", optim.rep_voxel),
                            dtype=torch.float32, device=device))


def case_seed(base, scene, material):
    """A fixed per-case sampler seed."""
    return int(base) + zlib.crc32(f"{scene}|{material}".encode()) % 10000


# ---------------------------------------------------------------------------
# Environment parameterisations
# ---------------------------------------------------------------------------

class LatentEnv:
    """The prior_al mode: latent z (initialised at 0) and log-exposure through the decoder."""

    emitter = "mesh"

    def __init__(self, model, scale_init, device):
        self.model = model
        self.latent = torch.zeros(1, model.latent_dim, device=device, requires_grad=True)
        self.scale_log = torch.tensor(math.log(scale_init), device=device, requires_grad=True)

    def parameters(self):
        return [self.latent, self.scale_log]

    def state(self):
        return {"latent": self.latent.detach().clone(), "scale_log": self.scale_log.detach().clone()}

    def environment_at(self, state):
        """-> (radiance [H, W, 3], depth [H, W]) in linear, metric units."""
        decoded = LavalDataset.denormalize(self.model.decode(state["latent"]))
        radiance = decoded[0, :3].permute(1, 2, 0).clamp(0.0, RADIANCE_MAX)
        return radiance * torch.exp(state["scale_log"]), decoded[0, 3]

    def environment(self):
        return self.environment_at({"latent": self.latent, "scale_log": self.scale_log})

    def render(self, scene, params, radiance, depth, spp, seed):
        return render.render_shell(scene, params, radiance, shell_vertices(depth), spp=spp, seed=seed)


class PixelEnv:
    """The direct_al mode: free log-radiance and log-depth per pixel on the shell mesh."""

    emitter = "mesh"

    def __init__(self, resolution, radiance_init, depth_init, device):
        height, width = resolution
        self.radiance_log = torch.full((height, width, 3), math.log(radiance_init),
                                       device=device, requires_grad=True)
        self.depth_log = torch.full((height, width), math.log(depth_init),
                                    device=device, requires_grad=True)

    def parameters(self):
        return [self.radiance_log, self.depth_log]

    def state(self):
        return {"radiance_log": self.radiance_log.detach().clone(),
                "depth_log": self.depth_log.detach().clone()}

    def environment_at(self, state):
        return (torch.exp(state["radiance_log"]).clamp(0.0, RADIANCE_MAX),
                torch.exp(state["depth_log"]))

    def environment(self):
        return self.environment_at({"radiance_log": self.radiance_log, "depth_log": self.depth_log})

    def render(self, scene, params, radiance, depth, spp, seed):
        return render.render_shell(scene, params, radiance, shell_vertices(depth), spp=spp, seed=seed)


class EnvmapEnv(PixelEnv):
    """The direct_envmap mode: free log-radiance per pixel, depth fixed (not rendered)."""

    emitter = "envmap"

    def __init__(self, resolution, radiance_init, depth_init, device):
        super().__init__(resolution, radiance_init, depth_init, device)
        self.depth_log = self.depth_log.detach()

    def parameters(self):
        return [self.radiance_log]

    def render(self, scene, params, radiance, depth, spp, seed):
        return render.render_envmap(scene, params, render.wrap_columns(radiance), spp=spp, seed=seed)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def censor(rendered, target, sat_thresh, clamp_max):
    """Clamp the render to [0, 1] only where the 8-bit photograph is saturated (there the
    observation is censored, so a brighter render is not an error); elsewhere to `clamp_max`."""
    saturated = target.amax(dim=-1, keepdim=True) >= sat_thresh
    return torch.where(saturated, rendered.clamp(0.0, 1.0), rendered.clamp(0.0, clamp_max))


def shell_radius_towards(depth, points, eps=1e-8):
    """Radius of each point, and the shell's radius in that point's direction (bilinear)."""
    height, width = depth.shape
    radius = points.norm(dim=-1)
    direction = points / radius.clamp_min(eps).unsqueeze(-1)
    theta = torch.arccos(direction[:, 1].clamp(-1.0, 1.0))
    phi = torch.atan2(direction[:, 0], -direction[:, 2])

    fx = (phi / (2 * math.pi)) % 1.0 * width - 0.5
    fy = (theta / math.pi) * height - 0.5
    x0, y0 = torch.floor(fx), torch.floor(fy)
    wx, wy = fx - x0, fy - y0
    x0i = x0.long() % width
    x1i = (x0i + 1) % width
    y0i = y0.long().clamp(0, height - 1)
    y1i = (y0i + 1).clamp(0, height - 1)
    top = depth[y0i, x0i] * (1 - wx) + depth[y0i, x1i] * wx
    bottom = depth[y1i, x0i] * (1 - wx) + depth[y1i, x1i] * wx
    return radius, top * (1 - wy) + bottom * wy


def repulsion(depth, points, margin):
    """Keep the shell at least `margin` outside every object point: a hinge on the radial
    clearance while a point is enclosed, the bare distance once the shell has cut past it."""
    radius, shell_radius = shell_radius_towards(depth, points)
    gap = shell_radius - radius
    return torch.where(gap > 0, (margin - gap).clamp(min=0.0), -gap).mean()


def smoothness(field):
    """Sum of squared one-pixel differences, wrapping in both directions."""
    dx = field - torch.roll(field, 1, dims=1)
    dy = field - torch.roll(field, 1, dims=0)
    return (dx ** 2).sum() + (dy ** 2).sum()


def objective(env, scene, params, case, optim, spp, seed):
    """Render the current environment and evaluate the loss. Returns (loss, terms, images)."""
    radiance, depth = env.environment()
    rendered = env.render(scene, params, radiance, depth, spp, seed)

    pred = censor(rendered, case.target, optim.sat_thresh, optim.clamp_max)
    mask_sum = case.mask.sum()
    mse = (((pred - case.target) ** 2) * case.mask).sum() / mask_sum
    log_mse = (((torch.log(pred + 1e-6) - torch.log(case.target + 1e-6)) ** 2)
               * case.mask).sum() / mask_sum
    terms = {"image": mse, "image_log": optim.w_log_mse * log_mse}
    if optim.w_rep:
        terms["repulsion"] = optim.w_rep * repulsion(depth, case.points, optim.rep_margin)
    if optim.get("w_smooth"):
        field = torch.log(radiance + 1e-6) if optim.smooth_domain == "log" else radiance
        terms["smooth"] = optim.w_smooth * smoothness(field)
    if optim.get("w_depth_smooth"):
        field = torch.log(depth) if optim.smooth_domain == "log" else depth
        terms["depth_smooth"] = optim.w_depth_smooth * smoothness(field)
    loss = sum(terms.values())
    return loss, {k: v.item() for k, v in terms.items()}, (radiance, depth, rendered)


# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------

def plot_history(path, history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, values in history.items():
        if key != "grad_norm" and any(values):
            axes[0].plot(np.log10(np.maximum(values, 1e-12)), label=key)
    axes[0].set_title("log10 loss terms")
    axes[0].legend()
    axes[1].plot(history["grad_norm"])
    axes[1].set_yscale("log")
    axes[1].set_title("gradient norm")
    for ax in axes:
        ax.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_env_exr(path, radiance, depth):
    """The 4-channel environment: RGB radiance and metric depth."""
    pyexr.write(str(path), torch.cat([radiance, depth[..., None]], dim=-1).cpu().numpy())


def descend(env, scene, params, case, optim, save_dir, seed, desc):
    """Adam on `env`'s parameters; the iterate with the lowest loss is kept."""
    save_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(env.parameters(), lr=optim.lr, betas=tuple(optim.betas))
    history, best = {}, {"loss": float("inf")}

    for i in tqdm(range(optim.n_iters), desc=desc):
        optimizer.zero_grad()
        loss, terms, (radiance, depth, rendered) = objective(env, scene, params, case, optim,
                                                             optim.spp, seed + i)
        loss.backward()
        for tensor in env.parameters():
            tensor.grad.nan_to_num_(nan=0.0, posinf=1e-5, neginf=-1e-5)
        grad_norm = torch.nn.utils.clip_grad_norm_(env.parameters(), optim.grad_clip)

        terms["total"], terms["grad_norm"] = loss.item(), grad_norm.item()
        for key, value in terms.items():
            history.setdefault(key, []).append(value)
        if terms["total"] < best["loss"]:
            best = {"loss": terms["total"], "iter": i, "state": env.state()}
        optimizer.step()

        if optim.preview_every and i % optim.preview_every == 0:
            with torch.no_grad():
                cv2.imwrite(str(save_dir / f"render_{i}.png"),
                            tonemap(rendered.detach().cpu().numpy())[:, :, ::-1])
                write_env_exr(save_dir / f"env_{i}.exr", radiance.detach(), depth.detach())

    with torch.no_grad():
        radiance, depth = env.environment_at(best["state"])
    write_env_exr(save_dir / "best_env.exr", radiance, depth)
    torch.save({"iter": best["iter"], "loss": best["loss"],
                **{k: v.cpu() for k, v in best["state"].items()}}, save_dir / "best.pth")
    (save_dir / "history.json").write_text(json.dumps(history))
    plot_history(save_dir / "loss.png", history)
    return best


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("checkpoint", nargs="?",
                        help="run directory or .pth of the trained model (prior_al mode only)")
    parser.add_argument("--config", default="configs/optim.yaml")
    parser.add_argument("-o", "--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config entry, e.g. -o optim.n_iters=101")
    parser.add_argument("--name", help="run name under save_dir (default: mode, or the "
                                       "checkpoint's directory name for the prior_al mode)")
    parser.add_argument("--scenes", nargs="+", help="subset of scenes")
    parser.add_argument("--materials", nargs="+", help="subset of materials")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip cases that already have a best_env.exr")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if (args.mode == "prior_al") != (args.checkpoint is not None):
        parser.error("the prior_al mode takes a checkpoint; the direct modes take none")

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    device = torch.device(f"cuda:{args.gpu}")
    optim = cfg.optim
    if args.mode != "prior_al":
        optim = OmegaConf.merge(optim, cfg.direct_al)
    if args.mode == "direct_envmap":
        optim = OmegaConf.merge(optim, cfg.direct_envmap)
    emitter = MODES[args.mode]

    if args.mode == "prior_al":
        model = load_model(args.checkpoint, device)
        with torch.no_grad():
            env_shape = tuple(model.decode(torch.zeros(1, model.latent_dim, device=device)).shape[-2:])
        make_env = lambda: LatentEnv(model, optim.scale_init, device)
        ckpt = Path(args.checkpoint)
        name = args.name or (ckpt.name if ckpt.is_dir() else ckpt.parent.name)
    else:
        env_shape = tuple(optim.resolution)
        env_class = EnvmapEnv if args.mode == "direct_envmap" else PixelEnv
        make_env = lambda: env_class(env_shape, optim.scale_init, optim.depth, device)
        name = args.name or args.mode

    run_dir = Path(cfg.save_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({
        "mode": args.mode, "emitter": emitter, "checkpoint": args.checkpoint,
        "source_dir": cfg.source_dir, "materials": list(args.materials or cfg.materials),
        "optim": optim}), run_dir / "config.yaml")

    scenes = args.scenes or list_scenes(cfg)
    materials = args.materials or list(cfg.materials)
    cases = [(s, m) for s in scenes for m in materials]
    print(f"{args.mode}: {len(cases)} cases -> {run_dir}")
    for index, (scene, material) in enumerate(cases, 1):
        save_dir = run_dir / scene / material
        if args.skip_existing and (save_dir / "best_env.exr").exists():
            continue
        case = load_case(cfg.source_dir, scene, material, optim, device)
        scene_mi, params = render.load_scene(
            render.stage_xml(case.scene_dir, material, "input", emitter), emitter, env_shape)
        seed = case_seed(optim.seed, scene, material)
        torch.manual_seed(seed)
        best = descend(make_env(), scene_mi, params, case, optim, save_dir, seed,
                       desc=f"[{index}/{len(cases)}] {scene} | {material}")
        print(f"  best iterate {best['iter']}, loss {best['loss']:.5f}")
        render.release_scene(scene_mi, params, case)


if __name__ == "__main__":
    main()
