import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/AnchorScout_v21_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OFFSET = 24  
D_GATE = 8.5 

def get_row1_vector(img_gray, bg_templates):
    """6-bit DNA for Top Row (Slots 0-5)."""
    vector = []
    for col in range(6):
        cx, cy = int(74+(col*59.1)), 261
        roi = img_gray[cy+8:cy+18, cx-5:cx+5]
        diff = min([np.sum(cv2.absdiff(roi, bg[32:42, 19:29])) / 100 for bg in bg_templates])
        vector.append(1 if diff > D_GATE else 0)
    return vector

def is_player_at_row1_home(img_gray, first_ore_col):
    """Surgical check: Is player center exactly 24px left of the first ore center?"""
    target_x_center = int(74 + (first_ore_col * 59.1) - OFFSET)
    
    # Surgical ROI: Narrow band on Row 1 (Y: 240-280, X: +/- 15px around target)
    roi_x1 = max(0, target_x_center - 15)
    roi_x2 = min(img_gray.shape[1], target_x_center + 15)
    gutter_roi = img_gray[240:280, roi_x1:roi_x2]
    
    # Character fingerprint (Dark pixels < 60 intensity)
    dark_pixel_count = np.sum(gutter_roi < 60)
    return dark_pixel_count > 100 # Adjusted footprint threshold

def run_v21_surgical_scout():
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_last_trigger = 500 # Ensure immediate detection

    print(f"--- Running v2.1 Surgical Anchor Scout (Run_{TARGET_RUN}) ---")

    for i in range(len(buffer_files) - 1):
        frames_since_last_trigger += 1
        
        # Manually force the very first frame as the start of Floor 1
        if i == 0:
            floor_library.append({"floor": 1, "idx": 0})
            print(" [!] Floor 1: Forced Dataset Start")
            continue

        # Scan N+1 for the 'Start' signature
        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. Identify first ore in Top Row
        row1_vec = get_row1_vector(img_n1_gray, bg_t)
        try:
            first_ore_col = row1_vec.index(1)
        except ValueError:
            continue # Row 1 completely empty (noise/loading frame)

        # 2. Check for Surgical Anchor at N+1
        if is_player_at_row1_home(img_n1_gray, first_ore_col):
            # Only trigger if we aren't already mid-floor (100 frame lockout)
            if frames_since_last_trigger > 100:
                floor_num = len(floor_library) + 1
                
                # Pair with Frame N (The Handshake)
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                
                cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_n1, f"FRAME {i+1} (START FLOOR {floor_num})", (30, 50), 0, 0.7, (0,255,0), 2)
                
                output_path = os.path.join(OUTPUT_DIR, f"Anchor_{floor_num:03}.jpg")
                cv2.imwrite(output_path, np.hstack((bgr_n, bgr_n1)))
                
                floor_library.append({"floor": floor_num, "idx": i+1})
                print(f" [!] Boundary {floor_num} Confirmed: Frame {i} -> {i+1} (Player to left of Col {first_ore_col})")
                frames_since_last_trigger = 0

    print(f"\n[SUCCESS] v2.1 Scout mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v21_surgical_scout()