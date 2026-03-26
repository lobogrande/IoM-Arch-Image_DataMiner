import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import pandas as pd
import os

# --- FORENSIC CONFIGURATION ---
BUFFER_ROOT = cfg.get_buffer_path(0)
OUT_DIR = "sentinel_forensic_profiles"

# Ground Truth Ranges provided for profiling
BANNER_TARGETS = [
    {"name": "event_2013_double", "start": 2013, "end": 2065, "is_double": True},
    {"name": "event_2457_single", "start": 2457, "end": 2505, "is_double": False},
    {"name": "event_2886_single", "start": 2886, "end": 2926, "is_double": False},
    {"name": "event_3304_single", "start": 3304, "end": 3350, "is_double": False},
    {"name": "event_3735_single", "start": 3735, "end": 3782, "is_double": False}
]

# Search Window (Block Grid to Stage ROI)
Y_MIN, Y_MAX = 40, 480 

def get_banner_y(img_gray, is_double=False):
    """
    Finds the vertical center of the banner(s) using 
    Horizontal Projection Profiling (HPP).
    """
    # 1. Calculate row-wise mean intensity
    # Banners are significantly darker than the block grid background
    hpp = np.mean(img_gray, axis=1)
    
    # 2. Isolate the search zone
    zone = hpp[Y_MIN:Y_MAX]
    
    if is_double:
        # Find the two most significant local minima (valleys)
        # We expect them to be separated by ~40-60 pixels
        from scipy.signal import find_peaks
        # We invert HPP so valleys become peaks for the detector
        peaks, props = find_peaks(-zone, distance=30, prominence=10)
        sorted_peaks = sorted(peaks, key=lambda x: zone[x])
        # Return the Y-coords of the two darkest rows
        found_y = [p + Y_MIN for p in sorted_peaks[:2]]
        return sorted(found_y)
    else:
        # Find the single darkest row
        min_y = np.argmin(zone) + Y_MIN
        return [min_y]

def run_profiler():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    for target in BANNER_TARGETS:
        print(f"Profiling {target['name']}...")
        profile_data = []
        
        for f_idx in range(target['start'], target['end'] + 1):
            if f_idx >= len(all_files):
                break
                
            img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[f_idx]))
            if img_bgr is None:
                continue
            
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            y_coords = get_banner_y(img_gray, target['is_double'])
            
            # Capture data for each banner in the frame
            for i, y in enumerate(y_coords):
                intensity = np.mean(img_gray[y, :])
                profile_data.append({
                    "frame": f_idx,
                    "banner_idx": i,
                    "y_coord": y,
                    "intensity": intensity
                })
        
        # Post-process for velocity
        df = pd.DataFrame(profile_data)
        if not df.empty:
            # Calculate instantaneous velocity per banner index
            df = df.sort_values(['banner_idx', 'frame'])
            df['velocity'] = df.groupby('banner_idx')['y_coord'].diff().abs()
            
            # Output forensic CSV
            out_path = os.path.join(OUT_DIR, f"{target['name']}_profile.csv")
            df.to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")

if __name__ == "__main__":
    # Ensure scipy is available for peak detection in double banners
    try:
        import scipy
        run_profiler()
    except ImportError:
        print("Error: Profiling script requires 'scipy' for double-banner peak detection.")
        print("Install via: pip install scipy")