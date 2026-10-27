"""SageMaker Processing step: score the trained model on the held-out test split.

Writes evaluation.json in the SageMaker Model Registry metrics format. The
pipeline's condition step reads multiclass_classification_metrics.accuracy.value.
"""

import json
import os
import sys
import tarfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Overridable for local runs; defaults are the Processing job mount points.
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/processing/model")
TEST_DIR = os.environ.get("TEST_DIR", "/opt/ml/processing/test")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/evaluation")


def main():
    with tarfile.open(os.path.join(MODEL_DIR, "model.tar.gz")) as tar:
        tar.extractall(MODEL_DIR, filter="data")

    # Reuse the exact preprocessing that the endpoint serves with.
    sys.path.insert(0, os.path.join(MODEL_DIR, "code"))
    from inference import model_fn

    artifacts = model_fn(MODEL_DIR)
    classes = artifacts["classes"]
    test_ds = datasets.ImageFolder(TEST_DIR, artifacts["transform"])
    if test_ds.classes != classes:
        raise RuntimeError("Test classes do not match the trained model's classes")

    n = len(classes)
    confusion = torch.zeros(n, n, dtype=torch.long)
    torch.set_num_threads(os.cpu_count() or 1)
    with torch.no_grad():
        for x, y in DataLoader(test_ds, batch_size=64, num_workers=os.cpu_count() or 1):
            pred = artifacts["model"](x).argmax(1)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t, p] += 1

    tp = confusion.diag().float()
    precision = tp / confusion.sum(0).clamp(min=1)
    recall = tp / confusion.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    accuracy = (tp.sum() / confusion.sum()).item()

    report = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "macro_f1": {"value": f1.mean().item(), "standard_deviation": "NaN"},
            "confusion_matrix": {
                classes[i]: {classes[j]: int(confusion[i, j]) for j in range(n)} for i in range(n)
            },
        },
        "per_class_f1": {classes[i]: round(f1[i].item(), 4) for i in range(n)},
        "test_images": int(confusion.sum()),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"accuracy={accuracy:.4f} macro_f1={f1.mean().item():.4f}")


if __name__ == "__main__":
    main()
