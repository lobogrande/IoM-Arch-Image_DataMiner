import cv2
import os

# --- 1. YOUR AUDITED FILENAMES & TARGET DIGITS ---
# We break these into individual digit crops to keep the logic consistent.
# Format: (filename, digit_value, x_offset_within_ROI, width_of_digit)
HARVEST_TASKS = [
    ("frame_20260306_231751_196638.png", 1, 0, 10),  # The '1' from Stage 10
    ("frame_20260306_231751_196638.png", 0, 12, 12), # The '0' from Stage 10
    ("frame_20260306_231751_741608.png", 1, 12, 10), # The second '1' from Stage 11 (with XP overlap)
    ("frame_20260306_231753_721292.png", 4, 12, 12), # The '4' from Stage 14 (the "ghost" 4)
    ("frame_20260306_231749_223602.png", 8, 0, 14)   # The '8' that looks like a 'B'
]

# Verified Coordinates
DIG_Y1, DIG_Y2, DIG_X1, DIG_X2 = 230, 246, 250, 281
DATASET_DIR = "capture_buffer_0"
DIGITS_DIR = "digits"

def run_precision_harvest():
    print(f"--- INITIATING PRECISION DIGIT HARVEST ---")
    
    for f_name, val, x_off, w in HARVEST_TASKS:
        img_path = os.path.join(DATASET_DIR, f_name)
        if not os.path.exists(img_path):
            print(f"!!! Error: Could not find {f_name}")
            continue
            
        img = cv2.imread(img_path, 0)
        # 1. Extract the full Number ROI
        roi = img[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2]
        
        # 2. Crop the specific digit within that ROI
        digit_crop = roi[0:16, x_off:x_off+w]
        
        # 3. Binarize at the "Gold Standard" 195
        _, binarized = cv2.threshold(digit_crop, 195, 255, cv2.THRESH_BINARY)
        
        # 4. Save using your naming convention
        # Format: {digit}_{modifier}_{id}.png
        out_name = f"{val}_noisy_{f_name.split('_')[-1]}"
        cv2.imwrite(os.path.join(DIGITS_DIR, out_name), binarized)
        print(f" Successfully harvested {val} from {f_name} -> {out_name}")

if __name__ == "__main__":
    run_precision_harvest()