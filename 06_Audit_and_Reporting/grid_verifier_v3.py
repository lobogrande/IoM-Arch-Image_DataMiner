import cv2
import numpy as np
import os
import random
import shutil

# --- 1. THE CALIBRATED CONSTANTS ---
DATASET_DIR = "capture_buffer_0"
OUTPUT_DIR = "grid_verification_report"
NUM_SAMPLES = 10

# Text ROIs
HEADER_ROI = (58, 105, 12, 22)      # Y, X, H, W
DIG_STAGE_ROI = (233, 163, 13, 116) # Y, X, H, W

# Grid Constants (Snapped for perfect alignment)
SLOT1_CENTER = (75, 261)
X_STEP, Y_STEP = 59.1, 59.1 # Perfectly square steps

def generate_verification():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    frames = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.png', '.jpg'))]
    samples = random.sample(frames, NUM_SAMPLES)
    
    for f_idx, f_name in enumerate(samples):
        img = cv2.imread(os.path.join(DATASET_DIR, f_name))
        if img is None: continue
        
        # A. HUD OVERLAY (Context Image)
        hud_img = img.copy()
        # Draw Header and Dig Stage HUDs
        cv2.rectangle(hud_img, (103, 56), (129, 72), (255, 0, 255), 1)
        cv2.rectangle(hud_img, (161, 231), (281, 248), (255, 0, 255), 1)
        
        # B. GRID EXTRACTION (AI Crops)
        # We will build a large composite of all 24 AI slots
        composite = np.zeros((48*4, 48*6, 3), dtype=np.uint8)
        
        for i in range(24):
            row, col = divmod(i, 6)
            cx = int(SLOT1_CENTER[0] + (col * X_STEP))
            cy = int(SLOT1_CENTER[1] + (row * Y_STEP))
            
            # Draw HUD Box
            cv2.rectangle(hud_img, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
            
            # Extract AI Crop (The 48x48 region)
            crop = img[cy-24:cy+24, cx-24:cx+24]
            if crop.shape[0] == 48 and crop.shape[1] == 48:
                composite[row*48:(row+1)*48, col*48:(col+1)*48] = crop
        
        # Save Outputs
        cv2.imwrite(f"{OUTPUT_DIR}/S{f_idx}_Full_HUD.jpg", hud_img)
        cv2.imwrite(f"{OUTPUT_DIR}/S{f_idx}_AI_Grid_Composite.jpg", composite)
        
        print(f"Generated verification for {f_name}")

if __name__ == "__main__":
    generate_verification()