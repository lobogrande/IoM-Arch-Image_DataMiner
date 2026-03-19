import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

def run_tuning_probe():
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170
    
    detailed_logs = []
    
    # Analyze the window frame-by-frame
    for i in range(target_idx - 5, target_idx + 30):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # KEY REFINEMENT: Look at the CENTER ROI (35% to 65% width)
        # This prevents the side-ores from diluting the nucleation signature
        w = img_gray.shape[1]
        center_roi = img_gray[:, int(w*0.35):int(w*0.65)]
        
        hpp_center = np.mean(center_roi, axis=1)
        var_center = np.var(center_roi, axis=1)
        
        # Find the "best" row in the mining area (y: 200-550)
        search_zone = slice(200, 550)
        best_y = np.argmin(hpp_center[search_zone]) + 200
        
        detailed_logs.append({
            "idx": i,
            "best_y": best_y,
            "center_intensity": hpp_center[best_y],
            "center_variance": var_center[best_y],
            "full_row_intensity": np.mean(img_gray[best_y, :]),
            "full_row_variance": np.var(img_gray[best_y, :])
        })

    df = pd.DataFrame(detailed_logs)
    df.to_csv("sentinel_tuning_data.csv", index=False)
    print("Tuning Probe complete. Compare 'center_intensity' vs 'full_row_intensity' in the CSV.")

if __name__ == "__main__":
    run_tuning_probe()