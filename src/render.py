"""Mitsuba glue shared by `fit.py` and `evaluate.py`.

Two kinds of emitter carry the fitted environment:

* `mesh`: the scene's `light_sphere` shape, a textured emitting shell whose vertices are
  the equirectangular depth map (`pano.shell_vertices`) and whose radiance texture is the
  RGB panorama. This is a near-field light: objects see parallax and distance falloff.
* `envmap`: the scene's `envmap` emitter, the same panorama at infinity.

A benchmark scene ships both variants: `<material>_<stage>.xml` (mesh) and
`<material>_envmap_<stage>.xml` (envmap), with identical cameras, objects and materials.
"""

import gc
import os

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

from pano import shell_topology, shell_vertices

_variant_ready = False


def ensure_variant():
    """Select the Mitsuba variant on first use (MI_VARIANT overrides `cuda_ad_rgb`)."""
    global _variant_ready
    if not _variant_ready:
        mi.set_variant(os.environ.get("MI_VARIANT", "cuda_ad_rgb"))
        mi.set_log_level(mi.LogLevel.Error)
        _variant_ready = True


def stage_xml(scene_dir, material, stage, emitter):
    infix = "_envmap" if emitter == "envmap" else ""
    return scene_dir / f"{material}{infix}_{stage}.xml"


def install_topology(params, height, width):
    """Replace the scene's shell with the H x W equirectangular shell topology."""
    faces, uv = shell_topology(height, width)
    params["light_sphere.vertex_positions"] = dr.ravel(
        shell_vertices(torch.ones(height, width)).numpy(), "C")
    params["light_sphere.faces"] = faces.astype(np.uint32).reshape(-1)
    params["light_sphere.vertex_texcoords"] = uv.reshape(-1)
    params.update()


def load_scene(xml_path, emitter, env_shape=None):
    """Load a scene; for the mesh emitter, install the shell topology of the environment.

    `env_shape` is the (H, W) of the environment map that will be put on the shell.
    """
    ensure_variant()
    scene = mi.load_file(str(xml_path))
    params = mi.traverse(scene)
    if emitter == "mesh":
        install_topology(params, *env_shape)
    return scene, params


def wrap_columns(radiance):
    """[H, W, 3] -> [H, W + 2, 3]: the layout of `envmap.data`, with one wrap-around
    column at each side so lookups across the seam interpolate correctly."""
    return torch.cat([radiance[:, -1:], radiance, radiance[:, :1]], dim=1)


@dr.wrap(source="torch", target="drjit")
def render_shell(scene, params, radiance, vertices, spp=64, seed=1):
    """Differentiable render with the environment on the shell mesh (torch in, torch out)."""
    params["light_sphere.emitter.radiance.data"] = radiance
    params["light_sphere.vertex_positions"] = dr.ravel(vertices, "C")
    params.update()
    return mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed + 1)


@dr.wrap(source="torch", target="drjit")
def render_envmap(scene, params, radiance, spp=64, seed=1):
    """Differentiable render with the environment at infinity; `radiance` column-wrapped."""
    params["envmap.data"] = radiance
    params.update()
    return mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed + 1)


def set_environment(params, radiance, depth, emitter):
    """Put a fixed environment (CPU tensors) onto a loaded scene's parameters."""
    if emitter == "envmap":
        params["envmap.data"] = wrap_columns(radiance).numpy()
    else:
        params["light_sphere.emitter.radiance.data"] = radiance.numpy()
        params["light_sphere.vertex_positions"] = dr.ravel(shell_vertices(depth).numpy(), "C")
    params.update()


def release_scene(*objects):
    """Drop a scene and its parameters before loading the next one (the meshes are large)."""
    del objects
    gc.collect()
    dr.sync_thread()


def render_environment(xml_path, emitter, radiance, depth, spp, seed):
    """Non-differentiable render of a fixed environment. Returns a [H, W, 3] float array."""
    scene, params = load_scene(xml_path, emitter, env_shape=tuple(radiance.shape[:2]))
    set_environment(params, radiance, depth, emitter)
    image = np.array(mi.render(scene, spp=int(spp), seed=int(seed)))[..., :3]
    release_scene(scene, params)
    return image


def render_scene_file(xml_path, spp, seed):
    """Render a scene exactly as its XML describes it (the ground truth)."""
    ensure_variant()
    scene = mi.load_file(str(xml_path))
    image = np.array(mi.render(scene, spp=int(spp), seed=int(seed)))[..., :3]
    release_scene(scene)
    return image
