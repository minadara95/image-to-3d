# Image to 3D

Convert a single image into a textured 3D mesh using AI depth estimation — with a browser-based 3D viewer.

## How It Works

1. **Depth estimation** — Runs [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) on your image to produce a per-pixel depth map
2. **Mesh generation** — Projects the depth map onto a dense 3D vertex grid and connects them into triangles
3. **Texture mapping** — UV-maps the original image onto the mesh
4. **Export** — Outputs standard `.obj` + `.mtl` + `_texture.png` files

## Web UI (recommended)

Start the Flask server and open the browser viewer:

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

- Drag & drop or click to upload an image
- Adjust depth scale and resolution
- Click **Convert to 3D** — the model appears in the viewer when ready
- **Orbit** (left-drag) · **Zoom** (scroll) · **Pan** (right-drag)
- Toggle wireframe or reset the camera with the toolbar buttons

> Model weights (~100 MB) are downloaded automatically on first run. GPU (CUDA) speeds up conversion but CPU works fine.

## Command-Line Usage

```bash
# Basic
python image_to_3d.py photo.jpg

# With options
python image_to_3d.py photo.jpg --output my_model --depth-scale 1.5 --resolution 256
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--output` | image stem | Base name for output files |
| `--depth-scale` | `1.0` | Z exaggeration — larger = deeper/taller model |
| `--resolution` | `512` | Grid density per axis — higher = more detail, larger file |
| `--no-texture` | off | Skip texture, use plain white material |

## Output Files

| File | Description |
|---|---|
| `<name>.obj` | 3D mesh |
| `<name>.mtl` | Material definition |
| `<name>_texture.png` | Texture (resized copy of input) |

Open the `.obj` in **Blender**, **MeshLab**, or **Windows 3D Viewer**. From Blender you can export to GLB, STL, FBX, and more.

## Tips

- **Portraits and objects** work best — clear foreground/background separation gives cleaner depth
- Raise `--depth-scale` for flat scenes (e.g. landscapes) to exaggerate elevation
- Lower resolution (e.g. `128`) for a quick preview; raise it (e.g. `512`) for final quality
- The model runs on CPU if no GPU is available — expect ~10–30 seconds per image
