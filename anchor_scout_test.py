import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v62_Recall"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SENSITIVITY CONSTANTS
MATCH_THRESHOLD = 0.82  # Aggressive seeking
OFFSET_X = 24           
D_GATE_LIVE = 6.0       # Catch even the faintest Dirt1 pixels

def get_slot_state(roi, bg_template):
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    return 2 if diff > D_GATE_LIVE else 0

def run_v62_recall_audit():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    # Force initial image
    bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), bgr_start)
    
    last_logged_row_dna = None

    print(f"--- Running v6.2 High-Recall Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 100 == 0:
            print(f" [Scanning] Frame {i} | Found: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. TEMPLATE SEARCH (Row 1 Focus)
        search_roi = img_n1_gray[230:310, 0:450]
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. RESOLVE ANCHOR
            player_center_x = max_loc[0] + 20
            target_col = round((player_center_x + OFFSET_X - 74) / 59.1)
            
            if 0 <= target_col < 6:
                # 3. SCAN ROW DNA
                row_dna = []
                for c in range(6):
                    cx, cy = int(74 + (c * 59.1)), 261
                    roi = img_n1_gray[cy-5:cy+5, cx-5:cx+5]
                    row_dna.append(get_slot_state(roi, bg_t[0]))

                # 4. THE CORE TEST
                # Is the player standing next to the LEFTMOST live ore?
                try:
                    first_live_col = row_dna.index(2)
                except ValueError:
                    first_live_col = -1 # No live ores found in row 1

                if target_col == first_live_col:
                    # 5. PERSISTENCE (Ignore if row is identical to last capture)
                    if row_dna != last_logged_row_dna:
                        floor_num = len(floor_library) + 1
                        last_logged_row_dna = row_dna
                        
                        bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                        bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                        
                        floor_library.append({"floor": floor_num, "idx": i+1})
                        print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Col {target_col} | Score {max_val:.3f}")

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v62_recall_audit()