import cv2
import os

# --- 1. CURRENT VERIFIED COORDINATES ---
DIG_Y1, DIG_Y2 = 230, 246
DIG_X1, DIG_X2 = 250, 281 
THRESH_VAL = 195 # The v39.0 "Gold Standard"

# Dataset to probe
DATASET_DIR = "capture_buffer_0"
OUTPUT_DIR = "diagnostic_phase_1"

def run_visual_audit():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # We only need to check the first 500 frames to see if the "Floor 1-10" area is visible
    frames = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(('.png', '.jpg'))])[:500]
    
    print(f"--- PHASE 1: VISUAL ROI AUDIT ---")
    for f_name in frames:
        img = cv2.imread(os.path.join(DATASET_DIR, f_name), 0)
        if img is None: continue
        
        # Crop the AI's "Eye"
        roi = img[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2]
        
        # Apply the Threshold
        _, binarized = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY)
        
        # Save both for comparison
        # (Using a very small size, so we'll upscale them for easier viewing)
        upscaled = cv2.resize(binarized, (0,0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{OUTPUT_DIR}/bin_{f_name}", upscaled)

    print(f"Audit Complete. Check '{OUTPUT_DIR}' to see if the digits are visible as white shapes.")

if __name__ == "__main__":
    run_visual_audit()