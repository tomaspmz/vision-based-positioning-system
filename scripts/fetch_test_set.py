"""
Fetch the 2024 temporal test set from Google Earth Engine.

Downloads ~100 224×224 px crops (10 m/px GSD) from 2024 Sentinel-2 imagery
for a random subset of the hexagons already mapped during training data
mining. This implements REQ-2.2.1 (temporal domain adaptation).

Usage:
    python fetch_test_set.py            # default 100 test images
    python fetch_test_set.py --n 50     # custom count
"""

import argparse
import os
import random
import glob

from dotenv import load_dotenv
import ee
import h3
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
GEE_PROJECT = os.environ["GEE_PROJECT"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_level7")
TEST_DIR = os.path.join(BASE_DIR, "data", "test_2024")


def discover_mapped_hexes():
    """Return the list of hex IDs that have training tiles on disk."""
    paths = glob.glob(os.path.join(RAW_DIR, "*.png"))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def init_gee():
    """Authenticate / initialise Google Earth Engine."""
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def build_2024_composite():
    """Median composite of 2024 Sentinel-2 over the study region."""
    min_lon, min_lat, max_lon, max_lat = 51.0, 24.0, 56.5, 26.5
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(roi)
    )


def fetch_one(hex_id, composite):
    """Download a single 224×224 crop centred on the hex.

    Buffer = 224 px × 10 m/px / 2 = 1120 m from centre.
    """
    center_lat, center_lon = h3.cell_to_latlng(hex_id)
    point = ee.Geometry.Point([center_lon, center_lat])
    region = point.buffer(1120).bounds()

    try:
        url = composite.getThumbURL({
            "dimensions": "224x224",
            "region": region,
            "format": "png",
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 3000,
        })
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BILINEAR)

        out_path = os.path.join(TEST_DIR, f"{hex_id}.png")
        img.save(out_path)
        return hex_id
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch 2024 test set")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of test images to download")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(TEST_DIR, exist_ok=True)

    # Discover what hexagons we already have training tiles for
    all_hexes = discover_mapped_hexes()
    print(f"[test] Found {len(all_hexes)} mapped hexagons on disk")

    # Sample a subset (or all if fewer available)
    random.seed(args.seed)
    n = min(args.n, len(all_hexes))
    selected = random.sample(all_hexes, n)
    print(f"[test] Sampling {n} hexagons for 2024 test set")

    # Skip already-downloaded test images
    already = {os.path.splitext(f)[0] for f in os.listdir(TEST_DIR)
               if f.endswith(".png")}
    to_fetch = [h for h in selected if h not in already]
    print(f"[test] {len(already)} already on disk, {len(to_fetch)} to fetch")

    if not to_fetch:
        print("[test] Nothing to do — test set already complete.")
        return

    # GEE setup
    print("[test] Initialising Google Earth Engine...")
    init_gee()
    composite = build_2024_composite()

    # Parallel download
    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, h, composite): h for h in to_fetch}
        for fut in tqdm(as_completed(futures), total=len(to_fetch),
                        desc="Downloading 2024 test tiles"):
            if fut.result() is not None:
                success += 1

    print(f"[test] Done — {success}/{len(to_fetch)} images saved to {TEST_DIR}")


if __name__ == "__main__":
    main()
