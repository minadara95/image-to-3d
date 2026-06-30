# Image to 3D

Convert a single image into a full 3D model using AI — with a browser-based interactive viewer.

## Two backends

| Backend | Quality | Install |
|---|---|---|
| **TripoSR** *(recommended)* | Full 360° mesh — back, sides and front all predicted by AI | Extra steps below |
| **Depth-map fallback** | Front face only, relief-style | Included in base requirements |

The app automatically uses TripoSR when installed, and falls back to the depth-map approach otherwise.

## Setup

### Base install (depth-map fallback)
```bash
pip install -r requirements.txt
```

### TripoSR install (full 3D — recommended)
```bash
pip install -r requirements.txt
pip install rembg
pip install git+https://github.com/VAST-AI-Research/TripoSR
```

> Model weights download automatically on first run (~1 GB for TripoSR, ~100 MB for depth model).
> GPU (CUDA) is strongly recommended for TripoSR — CPU works but takes several minutes per image.

## Web UI

```bash
python app.py
```

Open **http://localhost:5000**

- Drag & drop or click to upload an image
- Adjust **Depth scale** and **Resolution** (used by the fallback only)
- Click **Convert to 3D**
- The status bar shows which backend is being used
- **Auto-spin** — rotates the model so the 3D shape is immediately visible
- **Wireframe** — shows mesh topology
- **Reset view** — returns camera to default position
- **Orbit** (left-drag) · **Zoom** (scroll) · **Pan** (right-drag)

## Command-line (depth-map fallback only)

```bash
python image_to_3d.py photo.jpg [--output name] [--depth-scale 0.3] [--resolution 256]
```

## Output

| Backend | Files |
|---|---|
| TripoSR | `<job>.glb` — single file with geometry + vertex colours |
| Depth-map | `<job>.obj` + `<job>.mtl` + `<job>_texture.png` |

Open `.glb` or `.obj` in **Blender**, **MeshLab**, or **Windows 3D Viewer**.

## Tips

- TripoSR works best with a clear subject on a plain background
- For TripoSR: portrait/product-style shots with the subject centred give the best mesh
- For the depth-map fallback: keep **Depth scale at 0.10–0.30** to avoid distortion
- Lower **Resolution** (128) for a fast preview; raise (512) for final export
