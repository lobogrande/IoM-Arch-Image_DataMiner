import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v80_Global"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90
D_GATE = 6.0 # Sensitized for Dirt1 blocks

def get_total_grid_dna(img_gray):
    """Generates a 24-bit signature of the entire board state."""
    dna = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74 + (col * 59.1)), int(261 + (row * 59.1))
        roi = img_gray[cy-5:cy+5, cx-5:cx+5]
        # Using a mean-based signal to detect 'something vs nothing'
        dna.append(1 if np.mean(roi) > 65 else 0)
    return dna

def run_v80_global_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_x = -999
    last_logged_dna = None

    # Save Floor 1
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v8.0 Global Event Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. SEEK PLAYER
        search_roi = img_n1_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        current_at_anchor = False
        current_x = -999
        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            if any(abs(current_x - a) <= 4 for a in VALID_ANCHORS):
                current_at_anchor = True

        # 2. TRIGGER: Teleport detected (Snap to anchor)
        if current_at_anchor and (current_x != last_x):
            # 3. VERIFICATION: Did the grid actually reset?
            current_dna = get_total_grid_dna(img_n1_gray)
            
            if current_dna != last_logged_dna:
                floor_num = len(floor_library) + 1
                last_logged_dna = current_dna
                
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                
                # Visual Labels
                cv2.putText(bgr_n, f"F{i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_n1, f"F{i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                floor_library.append({"floor": floor_num, "idx": i+1})
                
                print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Anchor: {current_x}")

        last_x = current_x

    # Final Map Export
    with open(f"Run_0_FloorMap_v80.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v80_global_audit()