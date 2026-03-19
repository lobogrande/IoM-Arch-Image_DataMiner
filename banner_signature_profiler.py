import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
# Target frame you mentioned: "frame_20260306_231817_939420.png"
# We will look for this frame to find its index
TARGET_FILENAME = "frame_20260306_231817_939420.png"
WINDOW_BEFORE = 10
WINDOW_AFTER = 50

def analyze_frame(args):
    idx, img_path = args
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # HPP: Row intensity averages
    # Row Variance: Row flatness (Low variance = solid color / black rectangle)
    hpp = np.mean(img, axis=1)
    row_var = np.var(img, axis=1)
    
    return {"idx": idx, "hpp": hpp, "var": row_var}

def run_banner_diagnostic():
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try:
        target_idx = all_files.index(TARGET_FILENAME)
        print(f"Target found at Index: {target_idx}")
    except ValueError:
        print(f"Error: {TARGET_FILENAME} not found. Running on default range 1170.")
        target_idx = 1170 # Estimated Floor 17 start

    start = max(0, target_idx - WINDOW_BEFORE)
    end = min(len(all_files), target_idx + WINDOW_AFTER)
    
    tasks = [(i, os.path.join(BUFFER_ROOT, all_files[i])) for i in range(start, end)]
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(analyze_frame, tasks))
    results = [r for r in results if r is not None]

    # --- 1. SIGNATURE PLOTTING ---
    # Rows vs. Frames (Heatmaps)
    hpp_matrix = np.array([r['hpp'] for r in results])
    var_matrix = np.array([r['var'] for r in results])

    # Plot Intensity (HPP) Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(hpp_matrix.T, aspect='auto', cmap='gray', origin='upper')
    plt.colorbar(label='Avg Brightness')
    plt.title('Banner Signature: Intensity Heatmap (HPP)')
    plt.ylabel('Row Y-Coordinate')
    plt.xlabel('Frame Count')
    plt.savefig('banner_intensity_signature.png')

    # Plot Variance Heatmap (Shows the "Flatness" of the black rectangle)
    plt.figure(figsize=(10, 8))
    plt.imshow(var_matrix.T, aspect='auto', cmap='hot', origin='upper')
    plt.colorbar(label='Pixel Variance')
    plt.title('Banner Signature: Variance Heatmap (Flatness)')
    plt.ylabel('Row Y-Coordinate')
    plt.xlabel('Frame Count')
    plt.savefig('banner_variance_signature.png')

    # --- 2. LOGGING THE PROFILE ---
    # We track the "Nucleation Point" - the darkest, flattest row in the target zone
    profile_data = []
    for r in results:
        # Focus on the ore grid (y: 200 to 550)
        grid_zone = r['hpp'][200:550]
        min_y = np.argmin(grid_zone) + 200
        profile_data.append({
            "idx": r['idx'],
            "center_y": min_y,
            "min_intensity": r['hpp'][min_y],
            "row_variance": r['var'][min_y]
        })
    
    pd.DataFrame(profile_data).to_csv("banner_forensic_signature.csv", index=False)
    print("Forensic signature profiling complete. Check signature PNGs.")

if __name__ == "__main__":
    run_banner_diagnostic()