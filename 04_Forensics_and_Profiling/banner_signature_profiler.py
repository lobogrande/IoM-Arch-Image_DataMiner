import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- SEARCH CONSTRAINTS (Calibrated from your feedback) ---
SCAN_Y_START = 40   
SCAN_Y_END = 450     
BANNER_HEIGHT = 45   # Average height of the banner box

def run_pathfinding_probe():
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try:
        target_idx = all_files.index(TARGET_FILENAME)
        print(f"Target found at Index: {target_idx}")
    except ValueError:
        target_idx = 1170
        print(f"Target not found, starting at index {target_idx}")

    path_data = []
    
    # Analyze the window frame-by-frame
    # We broaden the range to catch the entire arrival and exit
    for i in range(target_idx - 30, target_idx + 80):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        
        # Focus on Center Strip (35%-65%)
        c1, c2 = int(w * 0.35), int(w * 0.65)
        center_strip = img_gray[:, c1:c2]
        
        # 1D Intensity profile
        intensities = np.mean(center_strip, axis=1)
        
        # SLIDING WINDOW: Find the darkest 45-pixel block
        best_avg = 255
        best_y_top = -1
        
        for y in range(SCAN_Y_START, SCAN_Y_END - BANNER_HEIGHT):
            window_avg = np.mean(intensities[y : y + BANNER_HEIGHT])
            if window_avg < best_avg:
                best_avg = window_avg
                best_y_top = y
        
        # Log the "Best Match" for this frame
        path_data.append({
            "idx": i,
            "filename": fname,
            "y_top": best_y_top,
            "y_center": best_y_top + (BANNER_HEIGHT // 2),
            "avg_intensity": best_avg,
            "variance": np.var(intensities[best_y_top : best_y_top + BANNER_HEIGHT])
        })

    df = pd.DataFrame(path_data)
    df.to_csv("banner_ground_truth_path.csv", index=False)
    print("Pathfinding Probe complete. Result saved to 'banner_ground_truth_path.csv'.")

if __name__ == "__main__":
    run_pathfinding_probe()