"""
image_to_3d.py

Two backends, tried in order:

1. TripoSR  — full 360° 3D reconstruction (back/sides predicted by AI).
              Requires:  pip install trimesh rembg
                         pip install git+https://github.com/VAST-AI-Research/TripoSR
              Output: GLB with vertex colours.

2. Depth map fallback — flat relief from a depth estimation model.
              Requires:  pip install torch transformers scipy pillow numpy
              Output: OBJ + MTL + texture PNG.

CLI usage (depth-map fallback only):
    python image_to_3d.py photo.jpg [--output name] [--depth-scale 0.3] [--resolution 256]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Backend 1 — TripoSR (full 3D)
# ---------------------------------------------------------------------------

def _ensure_triposr_path():
    """Add the bundled TripoSR source to sys.path so `tsr` can be imported."""
    triposr_src = Path(__file__).parent / "_triposr_src"
    if triposr_src.exists() and str(triposr_src) not in sys.path:
        sys.path.insert(0, str(triposr_src))


def run_triposr(image: Image.Image, out_glb: Path, resolution: int = 256,
                progress_cb=None) -> None:
    """
    Generate a full 3D mesh with TripoSR and export it as a GLB file.

    Raises ImportError if TripoSR / rembg are not installed.
    """
    _ensure_triposr_path()
    import torch
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if progress_cb: progress_cb("Loading TripoSR model…")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to(device)
    model.eval()

    if progress_cb: progress_cb("Removing background…")
    processed = remove_background(image)
    processed = resize_foreground(processed, 0.85)

    if progress_cb: progress_cb("Generating 3D geometry…")
    with torch.no_grad():
        scene_codes = model([processed], device=device)

    if progress_cb: progress_cb("Extracting mesh…")
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution, threshold=25.0)

    mesh = meshes[0]
    mesh.export(str(out_glb))
    print(f"Saved GLB → {out_glb}")


# ---------------------------------------------------------------------------
# Backend 2 — Depth-map fallback (front face only)
# ---------------------------------------------------------------------------

def estimate_depth(image: Image.Image) -> np.ndarray:
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit("Missing dependency: pip install transformers torch")

    print("Loading depth model (Depth Anything V2 Small) …")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    print("Running depth estimation …")
    result = pipe(image)
    depth = np.array(result["depth"], dtype=np.float32)

    p2, p98 = np.percentile(depth, 2), np.percentile(depth, 98)
    depth = np.clip(depth, p2, p98)
    if p98 > p2:
        depth = (depth - p2) / (p98 - p2)

    try:
        from scipy.ndimage import gaussian_filter
        depth = gaussian_filter(depth, sigma=1.5)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
    except ImportError:
        pass

    return depth


def build_mesh(depth, texture_image, depth_scale=0.3, resolution=256):
    h, w = depth.shape
    step = max(1, max(h, w) // resolution)
    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    nr, nc = len(rows), len(cols)
    print(f"Building {nr}×{nc} grid mesh …")

    aspect = h / w
    xs = np.linspace(-1.0, 1.0, nc)
    ys = np.linspace(aspect, -aspect, nr)
    zs = depth[np.ix_(rows, cols)] * depth_scale

    vertices, uvs = [], []
    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            vertices.append((xs[ci], ys[ri], float(zs[ri, ci])))
            uvs.append((col / (w - 1), 1.0 - row / (h - 1)))

    normals = [(0.0, 0.0, 1.0)]
    faces = []
    SKIP = 0.12 * depth_scale

    def idx(ri, ci): return ri * nc + ci + 1

    for ri in range(nr - 1):
        for ci in range(nc - 1):
            corners = [float(zs[ri,ci]), float(zs[ri,ci+1]),
                       float(zs[ri+1,ci]), float(zs[ri+1,ci+1])]
            if max(corners) - min(corners) > SKIP:
                continue
            tl, tr = idx(ri,ci), idx(ri,ci+1)
            bl, br = idx(ri+1,ci), idx(ri+1,ci+1)
            faces.append(((tl,tl,1),(bl,bl,1),(tr,tr,1)))
            faces.append(((tr,tr,1),(bl,bl,1),(br,br,1)))

    return vertices, uvs, normals, faces


def write_obj(path, vertices, uvs, normals, faces, mtl_name):
    with open(path, "w") as f:
        f.write(f"# image_to_3d.py\nmtllib {mtl_name}\n\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("\nusemtl image_texture\n")
        for tri in faces:
            f.write("f " + " ".join(f"{v}/{t}/{n}" for v,t,n in tri) + "\n")


def write_mtl(path, texture_filename):
    with open(path, "w") as f:
        f.write("newmtl image_texture\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
        f.write(f"map_Kd {texture_filename}\n")


def run_depthmap(image: Image.Image, out_obj: Path, depth_scale: float,
                 resolution: int, progress_cb=None) -> None:
    if progress_cb: progress_cb("Running depth estimation…")
    depth = estimate_depth(image)

    if progress_cb: progress_cb("Building mesh…")
    verts, uvs, normals, faces = build_mesh(depth, image, depth_scale, resolution)

    mtl_path     = out_obj.with_suffix(".mtl")
    texture_path = out_obj.parent / (out_obj.stem + "_texture.png")
    image.save(texture_path)
    write_mtl(mtl_path, texture_path.name)
    write_obj(out_obj, verts, uvs, normals, faces, mtl_path.name)
    print(f"Saved OBJ → {out_obj}")


# ---------------------------------------------------------------------------
# CLI (depth-map only)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--depth-scale", type=float, default=0.3)
    parser.add_argument("--resolution",  type=int,   default=256)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Not found: {img_path}")

    stem    = args.output or img_path.stem
    out_obj = img_path.parent / f"{stem}.obj"
    image   = Image.open(img_path).convert("RGB")
    run_depthmap(image, out_obj, args.depth_scale, args.resolution,
                 progress_cb=print)
    print("Done.")


if __name__ == "__main__":
    main()
