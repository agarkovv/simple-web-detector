import json
import os

import requests
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from prometheus_client import Counter
from prometheus_flask_exporter import PrometheusMetrics
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from flask_cors import CORS

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)
CORS(app)

metrics = PrometheusMetrics(app)

print("started server")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()
preprocess = weights.transforms()

print("Loaded model")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@metrics.counter("app_http_inference_count", "number of http requests")
def predict():
    try:
        data = request.get_json(force=True)
        url = data.get("url") if isinstance(data, dict) else None
        if not url:
            return jsonify({"error": "Missing 'url' field"}), 400

        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        img = Image.open(resp.raw)

        batch = [preprocess(img)]
        prediction = model(batch)[0]
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]

        return jsonify({"objects": labels})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
