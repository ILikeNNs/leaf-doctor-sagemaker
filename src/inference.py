"""SageMaker PyTorch serving handlers (packaged into model.tar.gz under code/).

Request:  raw image bytes, Content-Type image/jpeg | image/png | application/x-image
Response: {"predictions": [{"label", "crop", "condition", "probability"}, ...]}
"""

import io
import json
import os

import torch
from PIL import Image
from torchvision import transforms

IMAGE_TYPES = {"image/jpeg", "image/png", "application/x-image"}
TOP_K = int(os.environ.get("TOP_K", "3"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch.set_num_threads(max(1, os.cpu_count() or 1))


def eval_transform(size):
    """Shared with train.py's validation loader, so serving matches training."""
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def model_fn(model_dir):
    with open(os.path.join(model_dir, "classes.json")) as f:
        meta = json.load(f)
    model = torch.jit.load(os.path.join(model_dir, "model.pt"), map_location="cpu").eval()
    return {"model": model, "classes": meta["classes"], "transform": eval_transform(meta["img_size"])}


def input_fn(body, content_type):
    if content_type not in IMAGE_TYPES:
        raise ValueError(f"Unsupported content type {content_type}; send raw image bytes")
    return Image.open(io.BytesIO(body)).convert("RGB")


@torch.no_grad()
def predict_fn(image, artifacts):
    x = artifacts["transform"](image).unsqueeze(0)
    probs = torch.softmax(artifacts["model"](x), dim=1)[0]
    top = torch.topk(probs, k=min(TOP_K, probs.numel()))
    results = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        label = artifacts["classes"][idx]
        crop, _, condition = label.partition("___")
        results.append({
            "label": label,
            "crop": crop.replace("_", " "),
            "condition": condition.replace("_", " ").strip(),
            "probability": round(p, 4),
        })
    return {"predictions": results}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
