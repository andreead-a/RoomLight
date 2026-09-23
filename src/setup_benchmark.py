#!/usr/bin/env python3
"""Download the benchmark and rebuild its derived files.

    python src/setup_benchmark.py data/benchmark

The benchmark is fetched from the `synthetic_benchmark/` folder of the Hugging Face repo
https://huggingface.co/andreead-a/RoomLight (about 200 MB: scenes/, meshes/, scenes.txt
and LICENSE) into the given directory, unless scenes/ and meshes/ are already there.

The released benchmark ships only what cannot be recomputed. Two things per scene are
rebuilt here, both exactly determined by files that are shipped:

    <scene>.obj         the emissive mesh: the equirectangular grid of
                        <scene>_depth.exr, with the panorama as its radiance texture.
                        This is what the scene XMLs load as `light_sphere`, and it is
                        the same 2.5D area light parameterisation `fit.py` optimises.

    objects_input.ply   the surface points of the two inserted objects of the `input`
                        view, in world space: the meshes under the transforms written in
                        that view's scene XML. `fit.py` reads it for the repulsion term
                        that keeps the fitted 2.5D area light outside the objects.
"""

import argparse
import io
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyexr
import torch
import trimesh
from huggingface_hub import snapshot_download

from pano import shell_topology, shell_vertices

REPO_ID = "andreead-a/RoomLight"
REPO_FOLDER = "synthetic_benchmark"


def download_benchmark(root):
    """Fetch `REPO_FOLDER` of `REPO_ID` so that its contents end up directly in `root`.

    `snapshot_download` mirrors the repo layout, so the folder is fetched into a staging
    directory under `root` and its contents moved up afterwards. The staging directory
    also holds the hub's resume metadata, which is why it survives an interrupted run and
    is only removed once everything has been moved.
    """
    staging = root / ".hf_download"
    print(f"downloading {REPO_ID}/{REPO_FOLDER} to {root}")
    start = time.perf_counter()
    snapshot_download(REPO_ID, allow_patterns=[f"{REPO_FOLDER}/**"], local_dir=staging)
    fetched = staging / REPO_FOLDER
    if not fetched.is_dir():
        raise SystemExit(f"{REPO_ID} has no {REPO_FOLDER}/ folder")
    size = 0
    for item in sorted(fetched.iterdir()):
        target = root / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))
        size += sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) \
            if target.is_dir() else target.stat().st_size
    shutil.rmtree(staging)
    print(f"downloaded {size / 2**20:.0f} MB in {time.perf_counter() - start:.0f}s\n")


def read_depth(path):
    """<scene>_depth.exr -> (H, W) float32 metric depth."""
    depth = pyexr.read(str(path)).astype(np.float32)
    return depth[..., 0] if depth.ndim == 3 else depth


def obj_static_blocks(height, width):
    """The texture coordinates and faces of an H x W shell, as OBJ text.

    Both are constants of the grid, so they are built once and reused for every scene;
    only the vertex positions differ. The v coordinate is flipped relative to
    `shell_topology`, whose convention is Mitsuba's: Mitsuba's OBJ loader flips it back
    on load, so writing it unflipped would mirror the environment vertically.
    """
    faces, uv = shell_topology(height, width)
    buffer = io.StringIO()
    np.savetxt(buffer, np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=-1), fmt="vt %.8f %.8f")
    index = faces + 1  # OBJ indices are 1-based
    np.savetxt(buffer, np.stack([index, index], axis=-1).reshape(len(faces), 6),
               fmt="f %d/%d %d/%d %d/%d")
    return buffer.getvalue()


def write_shell_obj(path, depth, static_blocks):
    """The emitting shell for one scene: vertices from `depth`, topology from the grid."""
    vertices = shell_vertices(torch.from_numpy(depth)).numpy()
    buffer = io.StringIO()
    np.savetxt(buffer, vertices, fmt="v %.8f %.8f %.8f")
    path.write_text(buffer.getvalue() + static_blocks)


def object_points(scene_dir, meshes_dir, stage):
    """World-space vertices of the objects a stage's XML inserts, in the XML's own order.

    Any of the three materials describes the same geometry -- they differ only in which
    BSDF the shapes reference -- so the first in name order is read.
    """
    candidates = sorted(p for p in scene_dir.glob(f"*_{stage}.xml")
                        if "_envmap_" not in p.name)
    if not candidates:
        raise FileNotFoundError(f"no {stage} scene XML in {scene_dir}")
    xml = candidates[0]
    points = []
    for shape in ET.parse(xml).getroot().findall("shape"):
        filename = shape.find("string[@name='filename']")
        transform = shape.find("transform/matrix")
        if filename is None or transform is None:
            continue  # the shell emitter: it carries no to_world transform
        matrix = np.array([float(v) for v in transform.get("value").split()]).reshape(4, 4)
        mesh = trimesh.load(meshes_dir / Path(filename.get("value")).name, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        points.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
    if not points:
        raise RuntimeError(f"{xml} inserts no transformed objects")
    return np.concatenate(points).astype(np.float32)


def write_point_ply(path, points):
    """A binary little-endian PLY holding `points` and nothing else."""
    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(points, dtype="<f4").tobytes())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("root", type=Path,
                        help="the benchmark directory; downloaded here if not present")
    parser.add_argument("--scenes", nargs="+", help="only these scenes")
    parser.add_argument("--force", action="store_true", help="rebuild files that exist")
    parser.add_argument("--no-download", action="store_true",
                        help="never download; fail if the benchmark is not at root")
    args = parser.parse_args()

    scenes_dir, meshes_dir = args.root / "scenes", args.root / "meshes"
    if not (scenes_dir.is_dir() and meshes_dir.is_dir()):
        if args.no_download:
            raise SystemExit(f"{args.root} does not hold scenes/ and meshes/")
        args.root.mkdir(parents=True, exist_ok=True)
        download_benchmark(args.root)
    for path in (scenes_dir, meshes_dir):
        if not path.is_dir():
            raise SystemExit(f"{path} is missing -- is {args.root} the benchmark root?")

    scene_dirs = sorted(d for d in scenes_dir.iterdir()
                        if d.is_dir() and (not args.scenes or d.name in args.scenes))
    print(f"{len(scene_dirs)} scene(s) in {scenes_dir}")

    static_blocks, grid = None, None
    written = 0
    start = time.perf_counter()
    for index, scene_dir in enumerate(scene_dirs, 1):
        scene = scene_dir.name
        obj, ply = scene_dir / f"{scene}.obj", scene_dir / "objects_input.ply"
        todo = args.force or not obj.exists() or not ply.exists()
        if not todo:
            print(f"[{index}/{len(scene_dirs)}] {scene}: present")
            continue

        depth = read_depth(scene_dir / f"{scene}_depth.exr")
        if grid != depth.shape:  # only rebuilt if a scene is on another grid
            grid, static_blocks = depth.shape, obj_static_blocks(*depth.shape)
        write_shell_obj(obj, depth, static_blocks)
        write_point_ply(ply, object_points(scene_dir, meshes_dir, "input"))
        written += obj.stat().st_size + ply.stat().st_size
        print(f"[{index}/{len(scene_dirs)}] {scene}: {obj.name} "
              f"{obj.stat().st_size / 2**20:.1f} MB, {ply.name} "
              f"{ply.stat().st_size / 2**20:.1f} MB")

    print(f"\nwrote {written / 2**20:.0f} MB in {time.perf_counter() - start:.0f}s. "
          f"Point `source_dir` in configs/optim.yaml at {scenes_dir}")


if __name__ == "__main__":
    main()
