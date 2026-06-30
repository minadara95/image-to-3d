"""
image_to_3d.py — Convert a single image to a textured 3D mesh (OBJ + MTL + texture).

Usage:
    python image_to_3d.py <input_image> [--output <name>] [--depth-scale <float>]
                          [--resolution <int>] [--no-texture]

Example:
    python image_to_3d.py photo.jpg
    python image_to_3d.py photo.jpg --output my_model --depth-scale 1.5

Output files:
    <name>.obj  — 3D mesh
    <name>.mtl  — material definition
    <name>.png  — texture (copy of input, resized)

Requirements:
    pip install torch torchvision transformers pillow numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def estimate_depth(image: Image.Image) -> np.ndarray:
    """Return a normalized float32 depth map (0=far, 1=near) using Depth Anything V2."""
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
    depth_pil: Image.Image = result["depth"]

    depth = np.array(depth_pil, dtype=np.float32)

    # Percentile normalization — clips outliers so extreme pixels don't dominate
    p2, p98 = np.percentile(depth, 2), np.percentile(depth, 98)
    depth = np.clip(depth, p2, p98)
    if p98 > p2:
        depth = (depth - p2) / (p98 - p2)

    # Smooth to soften depth transitions
    try:
        from scipy.ndimage import gaussian_filter
        depth = gaussian_filter(depth, sigma=1.5)
        # Re-normalize after blur
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
    except ImportError:
        pass

    return depth


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def build_mesh(
    depth: np.ndarray,
    texture_image: Image.Image,
    depth_scale: float = 0.3,
    resolution: int = 256,
) -> tuple[list, list, list, list]:
    """
    Build a solid bas-relief mesh from a depth map.

    Produces a block with:
      - Front face: displaced by depth (textured)
      - Back face:  flat plane behind the front
      - Four side walls connecting front edges to back

    This looks like a sculpted physical plaque rather than a warped sheet.
    """
    h, w = depth.shape
    step = max(1, max(h, w) // resolution)
    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    nr, nc = len(rows), len(cols)
    print(f"Building {nr}×{nc} bas-relief mesh …")

    aspect = h / w
    xs = np.linspace(-1.0, 1.0, nc)
    ys = np.linspace(aspect, -aspect, nr)
    zs = depth[np.ix_(rows, cols)] * depth_scale  # front-face Z (0 = far, depth_scale = near)

    BACK_Z = -depth_scale * 0.5   # flat back face sits behind the deepest point

    # ── Vertices ──────────────────────────────────────────────────────────────
    # Block 0:  front face  (nr*nc vertices)
    # Block 1:  back face   (nr*nc vertices, same XY, flat Z = BACK_Z)
    vertices: list = []
    uvs:      list = []

    for ri in range(nr):
        for ci in range(nc):
            u = cols[ci] / max(w - 1, 1)
            v = 1.0 - rows[ri] / max(h - 1, 1)
            vertices.append((xs[ci], ys[ri], float(zs[ri, ci])))
            uvs.append((u, v))

    for ri in range(nr):
        for ci in range(nc):
            u = cols[ci] / max(w - 1, 1)
            v = 1.0 - rows[ri] / max(h - 1, 1)
            vertices.append((xs[ci], ys[ri], BACK_Z))
            uvs.append((u, v))

    normals = [(0.0, 0.0, 1.0)]

    # helpers — OBJ indices are 1-based
    def fi(ri, ci): return ri * nc + ci + 1
    def bi(ri, ci): return nr * nc + ri * nc + ci + 1

    faces: list = []

    # ── Front face ─────────────────────────────────────────────────────────────
    SKIP = 0.15 * depth_scale   # skip quads with steep depth jump (seam artifacts)
    for ri in range(nr - 1):
        for ci in range(nc - 1):
            corners = [float(zs[ri,ci]), float(zs[ri,ci+1]),
                       float(zs[ri+1,ci]), float(zs[ri+1,ci+1])]
            if max(corners) - min(corners) > SKIP:
                continue
            tl, tr = fi(ri,ci),   fi(ri,ci+1)
            bl, br = fi(ri+1,ci), fi(ri+1,ci+1)
            faces.append(((tl,tl,1),(bl,bl,1),(tr,tr,1)))
            faces.append(((tr,tr,1),(bl,bl,1),(br,br,1)))

    # ── Back face (reversed winding) ───────────────────────────────────────────
    for ri in range(nr - 1):
        for ci in range(nc - 1):
            tl, tr = bi(ri,ci),   bi(ri,ci+1)
            bl, br = bi(ri+1,ci), bi(ri+1,ci+1)
            faces.append(((tl,tl,1),(tr,tr,1),(bl,bl,1)))
            faces.append(((tr,tr,1),(br,br,1),(bl,bl,1)))

    # ── Side walls ─────────────────────────────────────────────────────────────
    # Top (ri=0)
    for ci in range(nc - 1):
        ftl, ftr = fi(0,ci), fi(0,ci+1)
        btl, btr = bi(0,ci), bi(0,ci+1)
        faces.append(((ftl,ftl,1),(btl,btl,1),(ftr,ftr,1)))
        faces.append(((ftr,ftr,1),(btl,btl,1),(btr,btr,1)))
    # Bottom (ri=nr-1)
    for ci in range(nc - 1):
        fbl, fbr = fi(nr-1,ci), fi(nr-1,ci+1)
        bbl, bbr = bi(nr-1,ci), bi(nr-1,ci+1)
        faces.append(((fbl,fbl,1),(fbr,fbr,1),(bbl,bbl,1)))
        faces.append(((fbr,fbr,1),(bbr,bbr,1),(bbl,bbl,1)))
    # Left (ci=0)
    for ri in range(nr - 1):
        ftl, fbl = fi(ri,0), fi(ri+1,0)
        btl, bbl = bi(ri,0), bi(ri+1,0)
        faces.append(((ftl,ftl,1),(fbl,fbl,1),(btl,btl,1)))
        faces.append(((fbl,fbl,1),(bbl,bbl,1),(btl,btl,1)))
    # Right (ci=nc-1)
    for ri in range(nr - 1):
        ftr, fbr = fi(ri,nc-1), fi(ri+1,nc-1)
        btr, bbr = bi(ri,nc-1), bi(ri+1,nc-1)
        faces.append(((ftr,ftr,1),(btr,btr,1),(fbr,fbr,1)))
        faces.append(((fbr,fbr,1),(btr,btr,1),(bbr,bbr,1)))

    return vertices, uvs, normals, faces


# ---------------------------------------------------------------------------
# OBJ / MTL export
# ---------------------------------------------------------------------------

def write_obj(
    path: Path,
    vertices: list,
    uvs: list,
    normals: list,
    faces: list,
    mtl_name: str,
) -> None:
    print(f"Writing {path} …")
    with open(path, "w") as f:
        f.write(f"# Generated by image_to_3d.py\n")
        f.write(f"mtllib {mtl_name}\n\n")

        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")

        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")

        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("\n")

        f.write("usemtl image_texture\n")
        for tri in faces:
            parts = " ".join(f"{vi}/{ti}/{ni}" for vi, ti, ni in tri)
            f.write(f"f {parts}\n")


def write_mtl(path: Path, texture_filename: str) -> None:
    print(f"Writing {path} …")
    with open(path, "w") as f:
        f.write("# Generated by image_to_3d.py\n")
        f.write("newmtl image_texture\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {texture_filename}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Convert an image to a 3D mesh.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--output", "-o", default=None,
                        help="Output base name (default: <image stem>)")
    parser.add_argument("--depth-scale", type=float, default=1.0,
                        help="Z exaggeration factor (default: 1.0)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Max mesh grid resolution per axis (default: 512)")
    parser.add_argument("--no-texture", action="store_true",
                        help="Skip writing texture; use solid material")
    return parser.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Error: file not found: {img_path}")

    stem = args.output if args.output else img_path.stem
    out_dir = img_path.parent

    obj_path     = out_dir / f"{stem}.obj"
    mtl_path     = out_dir / f"{stem}.mtl"
    texture_path = out_dir / f"{stem}_texture.png"

    # Load image
    print(f"Loading image: {img_path}")
    image = Image.open(img_path).convert("RGB")

    # Depth estimation
    depth = estimate_depth(image)

    # Resize texture to match depth map resolution for UV accuracy
    tex_w, tex_h = image.size
    texture = image.resize((tex_w, tex_h), Image.LANCZOS)

    # Build mesh
    verts, uvs, normals, faces = build_mesh(
        depth,
        texture,
        depth_scale=args.depth_scale,
        resolution=args.resolution,
    )

    # Write texture
    if not args.no_texture:
        print(f"Saving texture: {texture_path}")
        texture.save(texture_path)
        tex_filename = texture_path.name
    else:
        tex_filename = ""

    # Write MTL
    write_mtl(mtl_path, tex_filename)

    # Write OBJ
    write_obj(obj_path, verts, uvs, normals, faces, mtl_path.name)

    print()
    print("Done!")
    print(f"  Mesh    : {obj_path}")
    print(f"  Material: {mtl_path}")
    if not args.no_texture:
        print(f"  Texture : {texture_path}")
    print()
    print("Open the .obj file in Blender, MeshLab, or any 3D viewer.")


if __name__ == "__main__":
    main()
