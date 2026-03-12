"""
VBAPS Visualization — Predicted vs. Actual locations on a geographic scatter.

Can be run standalone after training:
    python visualize.py                             # uses best_model.pt
    python visualize.py --checkpoint path/to/model.pt

Or imported and called from train.py.
"""

import argparse
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / SSH
import matplotlib.pyplot as plt
import contextily as ctx
from PIL import Image
from pyproj import Transformer

from dataset import (build_class_map, hex_center_coords,
                     get_test_loader, get_val_transform,
                     MasterTileDataset, TEST_DIR, RAW_DIR)
from model import build_model
from utils import softmax_weighted_centroid, haversine_km

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")


# ──────────────────────────────────────────────────────────────────────
# Core: collect predictions for a loader
# Collect predictions and run inference

@torch.no_grad()
def collect_predictions(model, loader, coords, device, top_k=5):
    """Run inference on a loader, return arrays of true/pred coords + distances.

    Confidence is returned as *normalised certainty*: 1 minus the
    normalised entropy of the softmax distribution.  Values range
    from 0 (uniform / maximally uncertain) to 1 (all mass on one class).
    """
    model.eval()
    all_true = []
    all_pred = []
    all_conf = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)  # (N, C)

        # Argmax prediction
        preds = logits.argmax(dim=1).cpu().numpy()
        pred_xy = coords[preds]
        true_xy = coords[labels.numpy()]

        # Normalised certainty = 1 - H(p) / log(C)
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum(dim=1)       # (N,)
        max_entropy = np.log(probs.shape[1])
        certainty = (1.0 - entropy.cpu().numpy() / max_entropy)  # 0..1

        all_true.append(true_xy)
        all_pred.append(pred_xy)
        all_conf.append(certainty)

    true = np.concatenate(all_true, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    conf = np.concatenate(all_conf, axis=0)
    dists = haversine_km(true[:, 0], true[:, 1], pred[:, 0], pred[:, 1])

    return true, pred, conf, dists


# Lat/lon → Web Mercator helper (needed for contextily basemaps)
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _to_merc(lats, lons):
    """Return (x, y) in Web Mercator given arrays of latitudes and longitudes."""
    x, y = _transformer.transform(lons, lats)
    return x, y


def _add_basemap(ax, zoom="auto"):
    """Add an Esri satellite tile background to a Web Mercator axes."""
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery,
                        zoom=zoom, attribution_size=6)
    except Exception as e:
        print(f"[viz] Basemap unavailable ({e}); continuing without tiles.")


def plot_pred_vs_actual(true, pred, dists, title="Predicted vs. Actual Location",
                        save_path=None):
    """Scatter of true locations coloured by error, with lines to predictions."""
    fig, ax = plt.subplots(figsize=(12, 8))

    tx, ty = _to_merc(true[:, 0], true[:, 1])
    px, py = _to_merc(pred[:, 0], pred[:, 1])

    # Plot thin lines from true → pred
    for i in range(len(tx)):
        ax.plot([tx[i], px[i]], [ty[i], py[i]],
                color="white", alpha=0.4, linewidth=0.5)

    # True locations
    ax.scatter(tx, ty, c="steelblue", s=18, alpha=0.9,
               label="True", zorder=3, edgecolors="none")

    # Predicted locations coloured by distance error
    sc = ax.scatter(px, py, c=dists, cmap="RdYlGn_r",
                    s=18, alpha=0.9, label="Predicted", zorder=4,
                    edgecolors="none", vmin=0,
                    vmax=min(np.percentile(dists, 95), 50))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Error (km)")

    _add_basemap(ax)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{title}\nMean error: {np.mean(dists):.2f} km  |  "
                 f"Median: {np.median(dists):.2f} km")
    ax.legend(loc="upper left", facecolor="white", framealpha=0.7)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved: {save_path}")
    plt.close(fig)


def plot_error_histogram(dists, title="Localization Error Distribution",
                         save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    cap = min(np.percentile(dists, 99), 100)
    ax.hist(dists[dists <= cap], bins=50, color="steelblue", edgecolor="white",
            alpha=0.8)
    ax.axvline(np.mean(dists), color="red", linestyle="--",
               label=f"Mean = {np.mean(dists):.2f} km")
    ax.axvline(np.median(dists), color="orange", linestyle="--",
               label=f"Median = {np.median(dists):.2f} km")
    ax.set_xlabel("Error (km)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved: {save_path}")
    plt.close(fig)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def plot_prediction_grid(model, loader, coords, device, idx_to_hex,
                         raw_dir=None, n_pairs=8,
                         title="Sample Predictions — Input vs Predicted Location",
                         save_path=None):
    """4×4 grid of (input image | predicted-hex master tile) pairs.

    Rows: n_pairs // 2  (4 rows for 8 pairs)
    Cols: 4             (input | predicted | input | predicted)

    The input column shows the exact crop the model received (224×224).
    The predicted column shows the FULL master tile of the predicted hex,
    with a red/green border so the two columns are visually distinct.
    """
    if raw_dir is None:
        raw_dir = RAW_DIR

    model.eval()
    samples = []

    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        pred_classes = probs.argmax(dim=1).cpu().numpy()

        # Distance using argmax class centre
        preds_coords = coords[pred_classes]
        true_coords = coords[labels.numpy()]
        dists = haversine_km(true_coords[:, 0], true_coords[:, 1],
                             preds_coords[:, 0], preds_coords[:, 1])

        # Denormalize: (N, C, H, W) → (N, H, W, C), float in [0, 1]
        imgs_np = images.permute(0, 2, 3, 1).cpu().numpy()
        imgs_np = (imgs_np * _IMAGENET_STD + _IMAGENET_MEAN).clip(0.0, 1.0)

        for i in range(len(images)):
            pred_hex = idx_to_hex[int(pred_classes[i])]
            true_hex = idx_to_hex[int(labels[i])]
            samples.append((imgs_np[i], pred_hex, true_hex, float(dists[i])))

        if len(samples) >= n_pairs * 10:  # collect a pool, then sample randomly
            break

    chosen = random.sample(samples, min(n_pairs, len(samples)))

    nrows = (n_pairs + 1) // 2   # 4 rows for 8 pairs
    ncols = 4                     # input | pred | input | pred
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    fig.suptitle(title, fontsize=12, y=1.01)

    for i, (inp_img, pred_hex, true_hex, dist) in enumerate(chosen):
        row      = i // 2
        col_base = (i % 2) * 2   # 0 or 2

        ax_in   = axes[row, col_base]
        ax_pred = axes[row, col_base + 1]

        # Input crop (already denormalized, 224×224)
        ax_in.imshow(inp_img)
        ax_in.set_title(f"Input (224×224 crop)\n{true_hex[:12]}", fontsize=6.5)
        ax_in.axis("off")

        # FULL master tile for the predicted hex (512×512 displayed as-is)
        pred_path = os.path.join(raw_dir, f"{pred_hex}.png")
        if os.path.exists(pred_path):
            tile = np.array(Image.open(pred_path).convert("RGB"))
            ax_pred.imshow(tile)
        else:
            ax_pred.set_facecolor("#222")
            ax_pred.text(0.5, 0.5, "tile not found", ha="center", va="center",
                         color="white", fontsize=7, transform=ax_pred.transAxes)

        correct = pred_hex == true_hex
        label   = "✓ correct" if correct else f"{dist:.1f} km off"
        border_color = "green" if correct else "red"
        for spine in ax_pred.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_title(f"Predicted full tile ({label})\n{pred_hex[:12]}",
                          fontsize=6.5, color=border_color)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved: {save_path}")
    plt.close(fig)


def generate_all_plots(model, loader, coords, device, tag="val",
                       idx_to_hex=None):
    """Run inference + save all visualizations."""
    true, pred, conf, dists = collect_predictions(model, loader, coords, device)

    plot_pred_vs_actual(
        true, pred, dists,
        title=f"Predicted vs. Actual — {tag}",
        save_path=os.path.join(FIG_DIR, f"{tag}_pred_vs_actual.png"),
    )
    plot_error_histogram(
        dists,
        title=f"Localization Error — {tag}",
        save_path=os.path.join(FIG_DIR, f"{tag}_error_hist.png"),
    )
    if idx_to_hex is not None:
        plot_prediction_grid(
            model, loader, coords, device, idx_to_hex,
            title=f"Sample Predictions — {tag}",
            save_path=os.path.join(FIG_DIR, f"{tag}_prediction_grid.png"),
        )


# ──────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate VBAPS visualizations")
    parser.add_argument("--checkpoint",
                        default=os.path.join(BASE_DIR, "checkpoints", "best_model.pt"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--conf_threshold", type=float, default=0.20)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hex_to_idx = ckpt["hex_to_idx"]
    idx_to_hex = ckpt["idx_to_hex"]
    num_classes = ckpt["num_classes"]
    coords = hex_center_coords(idx_to_hex)

    model = build_model(num_classes=num_classes, freeze_backbone=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Val set (all tiles, centre crop)
    from torch.utils.data import DataLoader
    paths, hex_ids, _, _ = build_class_map()
    val_ds = MasterTileDataset(paths, hex_ids, hex_to_idx,
                               transform=get_val_transform(), random_crop=False)
    pin = torch.cuda.is_available()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"[viz] Generating val plots ({len(val_ds)} samples)...")
    generate_all_plots(model, val_loader, coords, device, tag="val",
                       conf_threshold=args.conf_threshold)

    # Test set (if available)
    if os.path.isdir(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        test_loader = get_test_loader(hex_to_idx, batch_size=args.batch_size,
                                      num_workers=args.num_workers)
        print(f"[viz] Generating test plots...")
        generate_all_plots(model, test_loader, coords, device, tag="test_2024",
                           conf_threshold=args.conf_threshold)

    print(f"[viz] All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
