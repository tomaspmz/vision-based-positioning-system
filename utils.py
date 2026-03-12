"""Shared utilities for VBAPS — no project-internal dependencies."""

import numpy as np
import torch


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two points (all in degrees). Returns km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def softmax_weighted_centroid(probs, coords, top_k=5):
    """
    Given softmax probabilities (N, C) and class centre coordinates (C, 2),
    return (N, 2) predicted [lat, lon] using the top-K weighted centroid.
    """
    topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
    topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True)

    coords_t = torch.tensor(coords, dtype=torch.float32, device=probs.device)
    gathered = coords_t[topk_idx]  # (N, K, 2)
    pred = (topk_weights.unsqueeze(-1) * gathered).sum(dim=1)
    return pred.cpu().numpy()


def topk_accuracy(output, target, topk=(1, 3, 5)):
    """Compute top-k accuracy for the given k values. Returns raw counts."""
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[k] = correct_k.item()
    return res
