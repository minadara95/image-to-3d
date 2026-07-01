"""
app.py — Flask web server for Image-to-3D.

Tries TripoSR first (full 360° mesh → GLB).
Falls back to depth-map approach (front face only → OBJ) if TripoSR is not installed.
"""

import os
import uuid
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

UPLOAD_DIR = Path("static/uploads")
OUTPUT_DIR = Path("static/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}


def _set(job_id, **kw):
    jobs[job_id].update(kw)


def run_conversion(job_id: str, img_path: Path, depth_scale: float, resolution: int):
    _set(job_id, status="running", message="Starting…")
    try:
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        out_base = OUTPUT_DIR / job_id

        # ── Try TripoSR ──────────────────────────────────────────────────────
        try:
            from image_to_3d import run_triposr
            glb_path = Path(str(out_base) + ".glb")

            def cb(msg): _set(job_id, message=msg)
            run_triposr(image, glb_path,
                        resolution=min(resolution, 256),
                        progress_cb=cb)

            _set(job_id,
                 status="done", message="Complete",
                 model=f"/static/outputs/{job_id}.glb",
                 model_type="glb")
            return

        except ImportError:
            _set(job_id, message="TripoSR not found — using depth-map fallback…")
        except RuntimeError as e:
            _set(job_id, message=f"TripoSR skipped ({e}) — using depth-map fallback…")

        # ── Depth-map fallback ───────────────────────────────────────────────
        from image_to_3d import run_depthmap
        obj_path = Path(str(out_base) + ".obj")
        mtl_path = Path(str(out_base) + ".mtl")
        tex_path = Path(str(out_base) + "_texture.png")

        def cb(msg): _set(job_id, message=msg)
        run_depthmap(image, obj_path, depth_scale, resolution, progress_cb=cb)

        _set(job_id,
             status="done", message="Complete (depth-map fallback)",
             model=f"/static/outputs/{job_id}.obj",
             mtl=f"/static/outputs/{job_id}.mtl",
             texture=f"/static/outputs/{job_id}_texture.png",
             model_type="obj")

    except Exception as e:
        _set(job_id, status="error", message=str(e))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    depth_scale = float(request.form.get("depth_scale", 0.3))
    resolution  = int(request.form.get("resolution", 256))

    job_id   = uuid.uuid4().hex
    ext      = Path(file.filename).suffix or ".jpg"
    img_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(img_path)

    jobs[job_id] = {"status": "pending", "message": "Queued"}
    t = threading.Thread(target=run_conversion,
                         args=(job_id, img_path, depth_scale, resolution),
                         daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(jobs[job_id])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
