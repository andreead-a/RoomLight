#!/usr/bin/env python3
"""Score fitted environments on the synthetic benchmark.

Every case directory written by `fit.py` holds `best_env.exr`. This re-renders that
environment, through the emitter the run was fitted with, on the two views of its scene:

    input   the view the environment was fitted to (bunny and armadillo)
    eval    the held-out view: a sphere, in the same material, at a different position

and scores each against the ground truth over the object pixels (see `metrics.py`):
PSNR on the tonemapped image, RMSE and RGB angular error in linear radiance. The linear
ground truth is rendered once per (scene, material, view) and cached in `--gt-dir`.

    python src/evaluate.py results/r6_ref results/direct_al results/direct_envmap
    python src/evaluate.py results/r6_ref --scenes bathroom --spp 1024

Per case it writes `<view>_render.exr`, `<view>_render.png` and `metrics.json`; per run
`summary.json` holds the means and standard deviations over cases.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyexr
from omegaconf import OmegaConf

import render
from metrics import masked_psnr, rgb_angular_error, rmse, tonemap

STAGES = ("input", "eval")
METRICS = ("psnr_obj", "rmse", "angular_error")
GT_SEED, PRED_SEED = 101, 7


def read_environment(path):
    """best_env.exr -> (radiance [H, W, 3], depth [H, W]) as CPU tensors."""
    import torch
    env = pyexr.read(str(path)).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(env[..., :3])), torch.from_numpy(env[..., 3])


def ground_truth(scene_dir, material, stage, gt_dir, spp):
    """Linear render of the ground-truth scene, cached on disk."""
    path = gt_dir / scene_dir.name / f"{material}_{stage}.exr"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        image = render.render_scene_file(render.stage_xml(scene_dir, material, stage, "mesh"),
                                         spp, GT_SEED)
        pyexr.write(str(path), image)
    return pyexr.read(str(path)).astype(np.float32)[..., :3]


def score_case(case_dir, scene_dir, material, emitter, gt_dir, spp):
    radiance, depth = read_environment(case_dir / "best_env.exr")
    metrics = {}
    for stage in STAGES:
        pred = render.render_environment(render.stage_xml(scene_dir, material, stage, emitter),
                                         emitter, radiance, depth, spp, PRED_SEED)
        pyexr.write(str(case_dir / f"{stage}_render.exr"), pred)
        pred_ldr = tonemap(pred)
        cv2.imwrite(str(case_dir / f"{stage}_render.png"), pred_ldr[:, :, ::-1])

        gt = ground_truth(scene_dir, material, stage, gt_dir, spp)
        gt_ldr = cv2.imread(str(scene_dir / f"{material}_{stage}_rendered.jpg"))[:, :, ::-1]
        mask = cv2.imread(str(scene_dir / f"{stage}_mask.png"))[:, :, 0] > 127
        metrics[stage] = {
            "psnr_obj": masked_psnr(pred_ldr / 255.0, gt_ldr / 255.0, mask),
            "rmse": rmse(pred, gt, mask),
            "angular_error": rgb_angular_error(pred, gt, mask),
        }
    return metrics


def summarise(records, materials):
    """Mean and standard deviation over cases, per (material, stage) and over all."""
    rows = []
    for material in list(materials) + (["all"] if len(materials) > 1 else []):
        subset = [r for r in records if material in ("all", r["material"])]
        for stage in STAGES:
            values = [r["metrics"][stage] for r in subset if stage in r["metrics"]]
            if not values:
                continue
            row = {"material": material, "stage": stage, "n": len(values)}
            for key in METRICS:
                row[key] = float(np.mean([v[key] for v in values]))
                row[key + "_std"] = float(np.std([v[key] for v in values]))
            rows.append(row)
    return rows


def print_table(run_name, rows):
    header = (f"{'material':<12} {'stage':<6} {'n':>3}"
              + "".join(f"  {label:>16}" for label in ("PSNR obj", "RMSE", "RGB angle")))
    print(f"\n{run_name}\n{header}\n{'-' * len(header)}")
    for row in rows:
        print(f"{row['material']:<12} {row['stage']:<6} {row['n']:>3}"
              + "".join(f"  {row[key]:>8.3f} +-{row[key + '_std']:>6.3f}" for key in METRICS))
    print("PSNR obj: higher is better; RMSE and RGB angular error (degrees): lower is better.\n"
          "'input' is the fitted view, 'eval' the held-out one.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories written by fit.py")
    parser.add_argument("--config", default="configs/optim.yaml")
    parser.add_argument("--spp", type=int, help="override evaluate.spp of the config")
    parser.add_argument("--gt-dir", type=Path,
                        help="cache of ground-truth renders (default: <save_dir>/_ground_truth)")
    parser.add_argument("--scenes", nargs="+", help="subset of scenes")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-render cases that already have a metrics.json")
    args = parser.parse_args()
    spp = args.spp or OmegaConf.load(args.config).evaluate.spp

    for run_dir in args.runs:
        run_cfg = OmegaConf.load(run_dir / "config.yaml")
        source_dir = Path(run_cfg.source_dir)
        gt_dir = args.gt_dir or run_dir.parent / "_ground_truth"
        cases = sorted(p.parent for p in run_dir.glob("*/*/best_env.exr")
                       if not args.scenes or p.parent.parent.name in args.scenes)
        records = []
        for case_dir in cases:
            scene, material = case_dir.parent.name, case_dir.name
            metrics_path = case_dir / "metrics.json"
            if metrics_path.exists() and not args.overwrite:
                records.append(json.loads(metrics_path.read_text()))
                continue
            metrics = score_case(case_dir, source_dir / scene, material, run_cfg.emitter,
                                 gt_dir, spp)
            record = {"run": run_dir.name, "scene": scene, "material": material,
                      "metrics": metrics}
            metrics_path.write_text(json.dumps(record, indent=2))
            records.append(record)
            print(f"  {scene} | {material}: " + "  ".join(
                f"{stage} psnr={m['psnr_obj']:.2f} rmse={m['rmse']:.3f} ang={m['angular_error']:.2f}"
                for stage, m in metrics.items()))

        rows = summarise(records, list(run_cfg.materials))
        (run_dir / "summary.json").write_text(json.dumps(
            {"run": run_dir.name, "spp": spp, "summary": rows, "per_case": records}, indent=2))
        print_table(run_dir.name, rows)


if __name__ == "__main__":
    main()
