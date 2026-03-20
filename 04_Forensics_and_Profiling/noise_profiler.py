import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/NoiseLab_v90"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_v90_noise_lab():
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # We'll analyze a 500-frame "Chaos Window"
    # Adjust the start_idx to a part of your video with lots of fairies/banners
    start_idx = 50 
    sample_size = 500
    
    frames = []
    print(f"--- Running v9.0 Noise Lab Profiler ---")
    print(f"Sampling {sample_size} frames for behavior analysis...")

    for i in range(start_idx, start_idx + sample_size):
        img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        if img is not None:
            frames.append(img)

    stack = np.stack(frames, axis=0)

    # 1. THE VARIANCE MAP (Where is the noise?)
    # High variance = pixels that change a lot.
    variance_map = np.var(stack, axis=0)
    variance_map = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. THE MAX-CONTRAST MAP (What is the "brightest" noise?)
    # This helps identify the exact shape of damage numbers and fairies
    max_map = np.max(stack, axis=0)
    
    # 3. TEMPORAL "GHOST" DIAGNOSTIC
    # We subtract Frame N from N+2 to see if 'slow' noise cancels out
    ghost_test = cv2.absdiff(frames[100], frames[102])
    
    # --- OUTPUT VISUALIZATION ---
    # Top: Max Map (See all noise at once)
    # Bottom: Variance Map (See where noise 'lives')
    combined = np.vstack((max_map, variance_map))
    
    # Draw horizontal markers for us to discuss the Y-coordinates
    for y in range(0, combined.shape[0], 50):
        cv2.line(combined, (0, y), (combined.shape[1], y), (255), 1)
        cv2.putText(combined, f"Y:{y}", (10, y-5), 0, 0.4, (255), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Master_Noise_Profile_{start_idx}_{sample_size}.jpg"), combined)
    print(f"\n[FINISH] Diagnostic map saved to {OUTPUT_DIR}/Master_Noise_Profile_{start_idx}_{sample_size}.jpg")

if __name__ == "__main__":
    run_v90_noise_lab()