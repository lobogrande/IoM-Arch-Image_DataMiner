import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v6_Elastic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SURGICAL CONSTANTS
MATCH_THRESHOLD = 0.88  # Slightly more elastic
OFFSET_X = 24           
D_GATE_LIVE = 6.5       # High sensitivity for Dirt1

def get_slot_state(roi, bg_template):
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    return 2 if diff > D_GATE_LIVE else 0

def run_v6_elastic_audit():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_logged_dna = None

    print(f"--- Running v6.0 Elastic Master Auditor ---")

    for i in range(len(buffer_files) - 1):
        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. SCAN FOR PLAYER (Full Row 1 Band)
        search_roi = img_n1_gray[230:310, 0:450] # Wide X-search
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. IDENTIFY NEAREST SLOT
            player_x = max_loc[0]
            # Calculate which column the player is "guarding"
            target_col = round((player_x + 20 + OFFSET_X - 74) / 59.1)
            
            if 0 <= target_col < 6:
                # 3. DNA STATE SCAN
                current_row_dna = []
                for c in range(6):
                    cx, cy = int(74 + (c * 59.1)), 261
                    roi = img_n1_gray[cy-5:cy+5, cx-5:cx+5]
                    current_row_dna.append(get_slot_state(roi, bg_t[0]))

                # 4. THE ELASTIC GATE
                # A: Target slot MUST have a live ore (2)
                # B: All slots to the LEFT MUST be empty (0)
                if current_row_dna[target_col] == 2:
                    left_gutter_clean = all(state == 0 for state in current_row_dna[:target_col])
                    
                    if left_gutter_clean:
                        # 5. PERSISTENCE CHECK (Prevent duplicates)
                        if current_row_dna != last_logged_dna:
                            floor_num = len(floor_library) + 1
                            last_logged_dna = current_row_dna
                            
                            # Log Handshake
                            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                            
                            floor_library.append({"floor": floor_num, "idx": i+1})
                            print(f" [!] Floor {floor_num} Found | Frame {i+1} | Col {target_col} | Score {max_val:.3f}")

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v6_elastic_audit()