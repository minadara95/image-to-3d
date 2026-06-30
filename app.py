"""
app.py — Flask web server for the Image-to-3D UI.

Usage:
    pip install flask
    python app.py
    Open http://localhost:5000
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

# Job state: job_id -> {"status": "pending"|"running"|"done"|"error", "message": str}
jobs: dict[str, dict] = {}


def run_conversion(job_id: str, img_path: Path, depth_scale: float, resolution: int):
    """Run depth estimation + mesh generation in a background thread."""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["message"] = "Running depth estimation…"

    try:
        from PIL import Image
        import numpy as np

        # Depth estimation
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        image = Image.open(img_path).convert("RGB")
        result = pipe(image)
        depth_pil = result["depth"]
        depth = np.array(depth_pil, dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)

        jobs[job_id]["message"] = "Building mesh…"

        # Import mesh builder from our existing module
        from image_to_3d import build_mesh, write_obj, write_mtl

        out_base = OUTPUT_DIR / job_id
        obj_path     = Path(str(out_base) + ".obj")
        mtl_path     = Path(str(out_base) + ".mtl")
        texture_path = Path(str(out_base) + "_texture.png")

        verts, uvs, normals, faces = build_mesh(
            depth, image,
            depth_scale=depth_scale,
            resolution=resolution,
        )

        image.save(texture_path)
        write_mtl(mtl_path, texture_path.name)
        write_obj(obj_path, verts, uvs, normals, faces, mtl_path.name)

        jobs[job_id]["status"]  = "done"
        jobs[job_id]["message"] = "Complete"
        jobs[job_id]["obj"]     = f"/static/outputs/{job_id}.obj"
        jobs[job_id]["mtl"]     = f"/static/outputs/{job_id}.mtl"
        jobs[job_id]["texture"] = f"/static/outputs/{job_id}_texture.png"

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    depth_scale = float(request.form.get("depth_scale", 0.3))
    resolution  = int(request.form.get("resolution",  256))

    # Save upload
    job_id   = uuid.uuid4().hex
    ext      = Path(file.filename).suffix or ".jpg"
    img_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(img_path)

    # Start background job
    jobs[job_id] = {"status": "pending", "message": "Queued"}
    t = threading.Thread(target=run_conversion, args=(job_id, img_path, depth_scale, resolution), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(jobs[job_id])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
