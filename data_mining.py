import ee
import h3
import pandas as pd
import requests
import os
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from skimage.measure import shannon_entropy
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
GEE_PROJECT = os.environ["GEE_PROJECT"]

# 1. Local Directory Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
local_data_dir = os.path.join(base_dir, 'data', 'raw_level7')
os.makedirs(local_data_dir, exist_ok=True)

# 2. Authenticate and Initialize Google Earth Engine
print("Authenticating with Google Earth Engine...")
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

# 3. Define the Coastal Bounding Box and Image Composite
min_lon, min_lat, max_lon, max_lat = 51.0, 24.0, 56.5, 26.5
roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

s2_composite = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(roi) \
    .filterDate('2022-01-01', '2022-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median() \
    .clip(roi)

# 4. Mathematically Tile the Bounding Box
print("Calculating unique H3 Level 7 grid cells...")
hex_set = set()
for lat in np.arange(min_lat, max_lat, 0.015):
    for lon in np.arange(min_lon, max_lon, 0.015):
        hex_id = h3.latlng_to_cell(lat, lon, 7) 
        hex_set.add(hex_id)

hex_list = list(hex_set)
print(f"Found {len(hex_list)} unique Level 7 hexagons in the region.")

# 5. The Worker Function
def process_hexagon(hex_id):
    center_lat, center_lon = h3.cell_to_latlng(hex_id) 
    point_geom = ee.Geometry.Point([center_lon, center_lat])
    patch_region = point_geom.buffer(2560).bounds()
    
    try:
        url = s2_composite.getThumbURL({
            'dimensions': '512x512',
            'region': patch_region,
            'format': 'png',
            'bands': ['B4', 'B3', 'B2'], 
            'min': 0,
            'max': 3000
        })
        
        # 10-second timeout prevents infinite hanging on bad network drops
        response = requests.get(url, timeout=10) 
        img = Image.open(BytesIO(response.content)).convert('L')
        img_array = np.array(img)
        
        img_entropy = shannon_entropy(img_array)
        if img_entropy >= 5.2: # Entropy threshold
            img_filename = f"{hex_id}.png"
            img_path = os.path.join(local_data_dir, img_filename)
            
            with open(img_path, 'wb') as fd:
                fd.write(response.content) 
                
            return {
                'image_id': img_filename,
                'h3_index': hex_id,
                'lat': center_lat,
                'lon': center_lon,
                'entropy_score': round(img_entropy, 2)
            }
    except Exception as e:
        # Silently fail for empty ocean tiles or API timeouts
        pass
    
    return None 

# 6. PARALLEL EXECUTION
print(f"Starting parallel extraction and filtering to local disk...")
data_log = []

# Note: Adjust max_workers if your home internet connection gets rate-limited by Google
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_hexagon, hex_id): hex_id for hex_id in hex_list}
    
    for future in tqdm(as_completed(futures), total=len(hex_list)):
        result = future.result()
        if result is not None:
            data_log.append(result)

# 7. Save the Master Label CSV locally
csv_path = os.path.join(base_dir, 'data', 'h3_labels_level7.csv')
df = pd.DataFrame(data_log)
df.to_csv(csv_path, index=False)

print(f"\nExtraction complete! Saved {len(data_log)} high-feature images to {local_data_dir}")