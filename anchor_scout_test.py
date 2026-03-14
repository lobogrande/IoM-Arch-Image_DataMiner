import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/AnchorScout_v31_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SURGICAL CONSTANTS (Based on your 40x60 templates)
MATCH_THRESHOLD = 0.90  # High precision for static sprite
OFFSET_X = 24           # Center-to-center offset
T_W, T_H = 40, 60       # Your extracted template dimensions

def run_v31_template_anchor():
    # 1. Load your newly harvested template
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    
    if player_right is None:
        print("Error: templates/player_right.png not found!")
        return

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    floor_library = []
    frames_since_trigger = 500

    print(f"--- Running v3.1 Template Anchor (Run_{TARGET_RUN}) ---")

    for i in range(len(buffer_files) - 1):
        frames_since_trigger += 1
        
        # Explicitly handle the "Dataset Start" (No side-by-side needed)
        if i == 0:
            floor_library.append({"floor": 1, "idx": 0})
            bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
            cv2.putText(bgr_start, "START OF RUN - FLOOR 1", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "START_Floor001.jpg"), bgr_start)
            continue

        # Load N+1 for the 'Start' signature
        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 2. TEMPLATE MATCH (Searching the Top Row / Left Gutter area)
        # Narrow Y band (200-350) and X gutter (0-150)
        search_roi = img_n1_gray[200:350, 0:150]
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # 3. VERIFICATION LOGIC
        if max_val > MATCH_THRESHOLD:
            # Player center X relative to the full screen
            # (max_loc[0] is X in ROI, + T_W//2 is center)
            player_center_x = max_loc[0] + (T_W // 2)
            
            # Calculate where the first ore SHOULD be
            expected_ore_x = player_center_x + OFFSET_X
            col = round((expected_ore_x - 74) / 59.1)
            
            if col in range(6) and frames_since_trigger > 60:
                # Surgical check: Is there actually an ore at that slot?
                cx, cy = int(74 + (col * 59.1)), 261
                slot_roi = img_n1_gray[cy-5:cy+5, cx-5:cx+5]
                diff = np.sum(cv2.absdiff(slot_roi, bg_t[0][19:29, 19:29])) / 100
                
                if diff > 8.5: # DNA confirms ore exists
                    floor_num = len(floor_library) + 1
                    
                    # Create the Handshake (Frame N vs Frame N+1)
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    
                    cv2.putText(bgr_n, f"FRAME {i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                    cv2.putText(bgr_n1, f"FRAME {i+1} (START FLOOR {floor_num})", (30, 50), 0, 0.7, (0,255,0), 2)
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Anchor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    
                    print(f" [!] Floor {floor_num} Confirmed! (Match: {max_val:.2f})")
                    frames_since_trigger = 0

    print(f"\n[SUCCESS] Mapped {len(floor_library)} floors using your surgical templates.")

if __name__ == "__main__":
    run_v31_template_anchor()