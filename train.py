"""
VBAPS Training Script — Frozen SeCo ResNet-50 classifier head.

Usage:
    python train.py                         # defaults
    python train.py --epochs 30 --lr 0.01   # custom
"""

import argparse
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm.auto import tqdm

from dataset import get_dataloaders, get_test_loader, hex_center_coords
from model import build_model
from utils import haversine_km, softmax_weighted_centroid, topk_accuracy
from visualize import generate_all_plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Training and validation loops

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, coords):
    model.eval()
    running_loss = 0.0
    total = 0
    topk_counts = {1: 0, 3: 0, 5: 0}
    all_dists_argmax = []

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        # Top-K accuracy
        counts = topk_accuracy(logits, labels, topk=(1, 3, 5))
        for k in counts:
            topk_counts[k] += counts[k]
        total += images.size(0)

        # Distance error — argmax (top-1 predicted class centre)
        preds = logits.argmax(dim=1).cpu().numpy()
        pred_coords = coords[preds]
        true_coords = coords[labels.cpu().numpy()]
        dists = haversine_km(
            true_coords[:, 0], true_coords[:, 1],
            pred_coords[:, 0], pred_coords[:, 1],
        )
        all_dists_argmax.extend(dists.tolist())

    # Random baseline: expected distance if predicting a random class
    num_classes = coords.shape[0]
    rng = np.random.default_rng(0)
    random_dists = []
    for _ in range(min(total, 2000)):
        i, j = rng.integers(0, num_classes, size=2)
        random_dists.append(haversine_km(
            coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]))
    random_baseline_km = float(np.mean(random_dists))

    metrics = {
        "loss": running_loss / total,
        "top1": topk_counts[1] / total,
        "top3": topk_counts[3] / total,
        "top5": topk_counts[5] / total,
        "mean_dist_km": float(np.mean(all_dists_argmax)),
        "median_dist_km": float(np.median(all_dists_argmax)),
        "random_baseline_km": random_baseline_km,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train VBAPS classifier head")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load best_model.pt and evaluate")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train] Using device: {device}")

    # Data
    train_loader, val_loader, meta = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    num_classes = meta["num_classes"]
    coords = meta["coords"]

    # Checkpoint path
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    # Model
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    model = model.to(device)

    # Loss (used in both training and eval)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.eval_only:
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No checkpoint found at {best_path}")
        print(f"[train] --eval-only: loading {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[train] Checkpoint from epoch {ckpt['epoch']}, "
              f"val top-1={ckpt['val_metrics']['top1']:.4f}")

        print("\n[train] Evaluating on validation set...")
        val_metrics = validate(model, val_loader, criterion, device, coords)
        print(f"  val    loss={val_metrics['loss']:.4f}  "
              f"top1={val_metrics['top1']:.4f}  "
              f"top3={val_metrics['top3']:.4f}  "
              f"top5={val_metrics['top5']:.4f}  "
              f"dist={val_metrics['mean_dist_km']:.2f} km  "
              f"(random baseline: {val_metrics['random_baseline_km']:.2f} km)")

        test_dir = os.path.join(BASE_DIR, "data", "test_2024")
        if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
            print("\n[train] Evaluating on 2024 test set...")
            test_loader = get_test_loader(
                meta["hex_to_idx"], batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            test_metrics = validate(model, test_loader, criterion, device, coords)
            print(f"  test   loss={test_metrics['loss']:.4f}  "
                  f"top1={test_metrics['top1']:.4f}  "
                  f"top3={test_metrics['top3']:.4f}  "
                  f"top5={test_metrics['top5']:.4f}  "
                  f"dist={test_metrics['mean_dist_km']:.2f} km  "
                  f"(random baseline: {test_metrics['random_baseline_km']:.2f} km)")
            print("\n[train] Generating test visualizations...")
            generate_all_plots(model, test_loader, coords, device,
                               tag="test_2024",
                               idx_to_hex=meta["idx_to_hex"])
        else:
            print("\n[train] No test set found.")

        print("\n[train] Generating validation visualizations...")
        generate_all_plots(model, val_loader, coords, device,
                           tag="val",
                           idx_to_hex=meta["idx_to_hex"])
        return

    # Training mode
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )

    # LR scheduler — cosine annealing over total epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    best_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.6f})")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_metrics = validate(model, val_loader, criterion, device, coords)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  val    loss={val_metrics['loss']:.4f}  "
              f"top1={val_metrics['top1']:.4f}  "
              f"top3={val_metrics['top3']:.4f}  "
              f"top5={val_metrics['top5']:.4f}  "
              f"dist={val_metrics['mean_dist_km']:.2f} km  "
              f"({elapsed:.1f}s)")

        # Save best checkpoint
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "num_classes": num_classes,
                "hex_to_idx": meta["hex_to_idx"],
                "idx_to_hex": meta["idx_to_hex"],
            }, best_path)
            print(f"  ** New best model saved (top1={best_top1:.4f})")

    print(f"\n[train] Finished. Best val top-1: {best_top1:.4f}")
    print(f"[train] Checkpoint: {best_path}")

    # Evaluate on 2024 test set if available
    test_dir = os.path.join(BASE_DIR, "data", "test_2024")
    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        print("\n[train] Evaluating on 2024 test set...")
        # Reload best weights
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        test_loader = get_test_loader(
            meta["hex_to_idx"], batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_metrics = validate(model, test_loader, criterion, device, coords)
        print(f"  test   loss={test_metrics['loss']:.4f}  "
              f"top1={test_metrics['top1']:.4f}  "
              f"top3={test_metrics['top3']:.4f}  "
              f"top5={test_metrics['top5']:.4f}  "
              f"dist={test_metrics['mean_dist_km']:.2f} km")

        # Generate test visualizations
        print("\n[train] Generating test visualizations...")
        generate_all_plots(model, test_loader, coords, device,
                           tag="test_2024",
                           idx_to_hex=meta["idx_to_hex"])
    else:
        print("\n[train] No test set found. Run fetch_test_set.py first to "
              "download the 2024 temporal test set.")

    print("\n[train] Generating validation visualizations...")
    generate_all_plots(model, val_loader, coords, device,
                       tag="val",
                       idx_to_hex=meta["idx_to_hex"])


if __name__ == "__main__":
    main()
