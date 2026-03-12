# Vision-Based Absolute Positioning System (VBAPS)
### Terrain Relative Navigation for GPS-Denied UAVs via Deep Learning

> **Iteration 1 — Frozen SeCo ResNet-50 Classifier Head**  
> Region: UAE/Qatar Coastline (Persian Gulf) · Imagery: Sentinel-2 · Year: 2022 train / 2024 test  
> 4,498 H3 hexagonal classes · Best checkpoint: epoch 180

---

## Overview

VBAPS is a Terrain Relative Navigation (TRN) prototype that estimates the **absolute geographic position of a UAV** using only a downward-facing camera feed — no GPS required. The system processes top-down satellite imagery and frames position estimation as a **multi-class geographic classification problem**, then recovers coordinates from the predicted class centre (argmax).

This repository contains the full pipeline: data acquisition from Google Earth Engine, H3 spatial indexing, model training, temporal-domain evaluation, and geographic visualizations with satellite basemaps.

---

## The Problem

GPS-denied navigation is a critical capability gap for UAVs operating in contested or degraded environments. Traditional TRN systems rely on pre-loaded terrain databases and laser altimetry. This project explores whether a deep learning model trained entirely on freely available satellite imagery can serve as a lightweight, passive positioning system using only a standard optical camera.

---

## Approach — Iteration 1

### 1. Geographic Classification (PlaNet Paradigm)

The core framing follows **PlaNet** (Weyand et al., 2016), which pioneered the now-standard approach of treating visual geolocation as classification over geographic cells rather than regression. The Earth (or in our case, the Persian Gulf coastline) is subdivided into discrete spatial buckets, and the model learns to classify which bucket a given image belongs to.

> *"We show that using a very large amount of geotagged photos we can train a probabilistic model to predict the location of any photo."*  
> — Weyand et al., **PlaNet - Photo Geolocation with Convolutional Neural Networks**, ECCV 2016

### 2. H3 Hexagonal Spatial Indexing

Rather than arbitrary rectangular grid cells (as used in PlaNet), we use **Uber's H3 hierarchical hexagonal indexing** at Level 7 (≈2.5 km diameter per cell). Hexagons have the key property that all neighbors are equidistant from the center, eliminating the corner-distance artifacts of square grids. This produces mutually exclusive, geographically coherent classification buckets that respect the continuous nature of space.

### 3. Entropy Filtering — Sparse Map

To prevent visual aliasing — where open ocean or featureless desert tiles are visually indistinguishable — we apply a **Shannon Entropy filter** (threshold ≥ 5.2 bits) to discard low-feature regions. This restricts the learnable map to structurally rich areas (coastlines, urban grids, islands, harbors), analogous to feature-point selection in classical SLAM.

### 4. SeCo Pretrained Backbone

The visual backbone is a **ResNet-50 pretrained with Seasonal Contrast (SeCo)** (Mañas et al., 2021), a self-supervised learning method specifically designed for **Sentinel-2 satellite imagery**. SeCo pretrains on 1 million geo-referenced satellite images from different seasons, learning features invariant to seasonal change — directly relevant to our temporal test strategy.

Using SeCo rather than ImageNet-pretrained weights is a deliberate design choice: CNN features trained on street-level photographs are suboptimal for overhead multispectral imagery. SeCo features already encode spectral and textural patterns specific to the Sentinel-2 sensor.

> *"We propose Seasonal Contrast (SeCo), an unsupervised pretraining strategy tailored to satellite imagery that makes use of images from the same geographic location captured at different times of year."*  
> — Mañas et al., **Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data**, ICCV 2021

In Iteration 1, the backbone is **fully frozen** — only the final classification head (`Linear(2048 → N_classes)`) is trained. This isolates the quality of SeCo's pretrained representations and provides a clean baseline before any task-specific fine-tuning.

### 5. UAV Feed Simulation — GSD Preservation

A critical implementation detail that separates this from naive approaches: the UAV camera footprint is simulated by extracting a **random 224×224 pixel crop** from a 512×512 master tile, with **no resizing**. This preserves the native **10 m/px Ground Sample Distance (GSD)** of the Sentinel-2 sensor, meaning the model always sees exactly a 2.24 km² footprint — the optical field of view of a UAV at the corresponding altitude.

Furthermore, each crop is extracted at a **random heading angle** (0–360°, continuous) using `cv2.warpAffine`, removing rotational invariance. Over thousands of training epochs, each master tile yields thousands of unique rotated perspectives — simulating the natural yaw variation of a UAV in flight.

### 6. Temporal Domain Adaptation

The training set uses **2022 Sentinel-2 imagery**. The test set is independently acquired from **2024 Sentinel-2 imagery** for the same geographic hexagons. This intentional temporal shift tests robustness to:
- Seasonal variation (vegetation, water levels)
- New construction (UAE develops extremely rapidly; palm islands, towers, reclaimed land)
- Atmospheric and illumination differences

Additionally, test images undergo a **random rotation** (0–360°, continuous) at inference time, verifying that the model has learned true rotational invariance from the random-heading crops used during training. This is a stricter evaluation than most academic geolocation benchmarks, which test on the same temporal distribution and canonical orientation as training.

### 7. Coordinate Recovery — Argmax Class Centre

The model outputs a probability distribution over all N hexagonal classes. The predicted geographic coordinate is the **centre of the top-1 predicted class** (argmax). Earlier experiments with softmax-weighted centroids over top-K classes were abandoned because spread-out runner-up classes produced centroids in physically impossible locations (e.g., the middle of the Persian Gulf).

### 8. Random Baseline Comparison

To validate that the model is learning meaningful spatial features, each evaluation reports a **random baseline**: the expected haversine distance if every prediction were a randomly chosen class. This provides an immediate sanity check — if the model's mean distance is well below the random baseline, it is learning, not memorising noise.

---

## Pipeline

```
Google Earth Engine (Sentinel-2 2022)
        │
        ▼
H3 Level-7 Tiling (~4,500 hexagons over Persian Gulf)
        │
        ▼
Shannon Entropy Filter (≥ 5.2) → Sparse Map (4,498 classes)
        │
        ▼
512×512 Master Tiles saved to disk (filename = hex_id)
        │
        ├─── Training ──→ Random rotated 224×224 crop → SeCo ResNet-50 (frozen) → fc head
        │
        └─── Testing  ──→ 2024 GEE imagery (same hexagons) + random rotation at inference
```

---

## Results — Iteration 1 (200 epochs, best at epoch 180)

| Metric | Validation | Test (2024) | Random Baseline |
|--------|-----------|-------------|-----------------|
| Top-1 | **44.3%** | **31.0%** | 0.02% |
| Top-3 | **66.1%** | **52.0%** | 0.07% |
| Top-5 | **75.2%** | **62.0%** | 0.11% |
| Mean Distance | **39.8 km** | **41.0 km** | ~206–222 km |
| Median Distance | — | — | — |

The model achieves **~5× lower positional error** than random guessing. Test set performance includes both temporal domain shift (2022→2024 imagery) and random rotation, confirming learned rotational invariance. The relatively small gap between validation and test metrics suggests the SeCo features generalise well across time.

---

## Evaluation Metrics

Two metrics are reported, each translating model output into a physically interpretable quantity:

**Top-K Accuracy** — Because geography is a continuum and adjacent hexagons look similar, Top-1 is a strict lower bound. Top-5 accuracy represents whether the model's uncertainty is geographically localised (all candidates are near the true location).

**Mean Haversine Distance (km)** — The primary aerospace metric. Uses the argmax class centre (not a centroid average) to compute kilometers of positional error via the haversine formula. A random baseline is reported alongside, computed as the expected distance between two randomly chosen class centres, providing a clear "is the model learning?" benchmark.

---

## Roadmap

| Iteration | Change | Expected Gain |
|-----------|--------|---------------|
| **1 (current)** | Frozen SeCo backbone, 200 epochs | Baseline |
| **2** | Unfreeze `layer4`, lower LR | −20–30 km distance |
| **3** | Full fine-tune | −10–15 km distance |
| **4** | ViT-Small backbone (CNN vs. Transformer comparison) | TBD |

---

## Technical Stack

| Component | Tool |
|-----------|------|
| Satellite imagery | Google Earth Engine — Sentinel-2 SR Harmonized |
| Spatial indexing | Uber H3 (Level 7) |
| Model | torchvision ResNet-50 + SeCo weights |
| Augmentation | Albumentations (train), OpenCV (rotated crops) |
| Training | PyTorch, SGD + Cosine Annealing LR |
| Visualization | Matplotlib + contextily (Esri satellite basemap) |
| Data parallel | MPS (Apple Silicon) / CUDA |

---

## References

1. **Weyand, T., Kostrikov, I., & Philbin, J.** (2016). *PlaNet - Photo Geolocation with Convolutional Neural Networks.* ECCV 2016. — Geographic cell classification paradigm, softmax weighted centroid.

2. **Mañas, O., Lacoste, A., Giró-i-Nieto, X., Karatzas, D., & Rodriguez, P.** (2021). *Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data.* ICCV 2021. — SeCo backbone pretrained weights (Zenodo: `seco_resnet50_1m.ckpt`).

3. **Müller, E., Zisserman, A., & Cummins, M.** (2018). *Geolocation Estimation of Photos Using a Hierarchical Model and Scene Classification.* ECCV 2018. — Hierarchical geographic classification and scene-adaptive localization.

4. **Uber Engineering.** (2018). *H3: Uber's Hexagonal Hierarchical Spatial Index.* — Hexagonal spatial indexing system used for geographic cell definition.

5. **European Space Agency.** *Sentinel-2 Mission.* Copernicus Programme. — Source of 10 m/px multispectral imagery used for both training (2022) and evaluation (2024).

6. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. — ResNet-50 backbone architecture.

---

## Usage

```bash
# 1. Acquire training data (2022 Sentinel-2)
python data_mining.py

# 2. Acquire test data (2024 Sentinel-2, same hexagons)
python fetch_test_set.py --n 100

# 3. Train (Iteration 1: frozen backbone, 200 epochs)
python train.py --epochs 200 --batch_size 32 --lr 0.01

# 4. Evaluate only (loads best checkpoint, runs val + test, generates plots)
python train.py --eval-only

# 5. Generate visualizations standalone
python visualize.py --checkpoint checkpoints/best_model.pt
```
