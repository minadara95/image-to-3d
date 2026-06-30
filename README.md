# Image to 3D

Convert a single image into a textured 3D mesh using AI depth estimation — with a browser-based interactive 3D viewer.

## How It Works

1. **Depth estimation** — Runs [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) on your image to produce a per-pixel depth map
2. **Depth processing** — Percentile normalization + Gaussian smoothing to reduce artifacts
3. **Mesh generation** — Projects depth onto a dense vertex grid; skips triangles at depth discontinuities to avoid stretched edges
4. **Texture mapping** — UV-maps the original image onto the mesh
5. **Export** — Outputs standard `.obj` + `.mtl` + `_texture.png` files

## Web UI (recommended)

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser.

- Drag & drop or click to upload an image
- Adjust **Depth scale** (0.10–1.50) and **Resolution** (64–512)
- Click **Convert to 3D** — model appears in the viewer when ready
- **Auto-spin** — toggles object rotation so the 3D shape is immediately visible
- **Wireframe** — shows the mesh topology
- **Reset view** — returns camera to the default position
- **Orbit** (left-drag) · **Zoom** (scroll) · **Pan** (right-drag)

> Model weights (~100 MB) download automatically on first run. GPU (CUDA) speeds up conversion but CPU works fine (~10–30 s/image).

## Command-Line Usage

```bash
# Basic
python image_to_3d.py photo.jpg

# With options
python image_to_3d.py photo.jpg --output my_model --depth-scale 0.3 --resolution 256
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--output` | image stem | Base name for output files |
| `--depth-scale` | `1.0` | Z exaggeration — larger = deeper relief |
| `--resolution` | `512` | Grid density — higher = more detail, larger file |
| `--no-texture` | off | Skip texture, use plain white material |

## Output Files

| File | Description |
|---|---|
| `<name>.obj` | 3D mesh |
| `<name>.mtl` | Material definition |
| `<name>_texture.png` | Texture (resized copy of input) |

Open the `.obj` in **Blender**, **MeshLab**, or **Windows 3D Viewer**. From Blender you can export to GLB, STL, FBX, and more.

## Tips

- **Depth scale 0.10–0.30** works best for most photos — keep it low to avoid distortion
- Images with a clear subject against a distinct background produce the cleanest mesh
- Lower resolution (128–256) for faster preview; raise to 512 for final export
- The **Auto-spin** button is the easiest way to see the 3D shape right after conversion
