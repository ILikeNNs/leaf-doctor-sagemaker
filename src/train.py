"""Fine-tune an ImageNet-pretrained CNN on PlantVillage (SageMaker training job).

Spot-friendly: a checkpoint is written to /opt/ml/checkpoints after every epoch
and training resumes from it if the instance is reclaimed.

Output (/opt/ml/model -> model.tar.gz):
    model.pt          TorchScript model (CPU) - no model class needed at inference
    classes.json      index -> class name
    code/inference.py SageMaker PyTorch serving handlers
"""

import argparse
import json
import os
import shutil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from inference import IMAGENET_MEAN, IMAGENET_STD, eval_transform

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints")  # override for local runs


def build_model(arch, num_classes):
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def loaders(train_dir, val_dir, img_size, batch_size, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, eval_transform(img_size))
    if train_ds.classes != val_ds.classes:
        raise RuntimeError("Train and val class lists differ")
    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        train_ds.classes,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--val", default=os.environ.get("SM_CHANNEL_VAL", "data/val"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = loaders(args.train, args.val, args.img_size, args.batch_size, args.workers)
    print(f"device={device} classes={len(classes)} train_batches={len(train_loader)}")

    model = build_model(args.arch, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch, best_acc = 0, -1.0  # -1 so the first epoch always saves best weights
    ckpt_path = os.path.join(CHECKPOINT_DIR, "last.pt")
    run_config = {"arch": args.arch, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                  "steps_per_epoch": len(train_loader)}
    ckpt = torch.load(ckpt_path, map_location=device) if os.path.exists(ckpt_path) else None
    if ckpt and ckpt.get("config") != run_config:
        print("Ignoring checkpoint from a run with a different configuration")
        ckpt = None
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        print(f"Resumed from spot checkpoint at epoch {start_epoch}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, "best_weights.pt")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
        val_acc = evaluate(model, val_loader, device)
        # Parsed by the training job's metric definitions / visible in CloudWatch.
        print(f"epoch={epoch} train_loss={running / len(train_loader):.4f} val_acc={val_acc:.4f} "
              f"time={time.time() - t0:.0f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_acc": best_acc, "config": run_config,
        }, ckpt_path)

    # Export the best epoch as a CPU TorchScript model for serverless inference.
    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to("cpu").eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, args.img_size, args.img_size))
    os.makedirs(args.model_dir, exist_ok=True)
    traced.save(os.path.join(args.model_dir, "model.pt"))
    with open(os.path.join(args.model_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "img_size": args.img_size, "arch": args.arch}, f)
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py"), code_dir)
    print(f"best_val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
