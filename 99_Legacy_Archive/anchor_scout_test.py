import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v97_FramePerfect"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.92

def get_stage_fingerprint(img_bgr):
    """Generates a binary state of the Stage Number text."""
    # ROI for 'Stage: [XXX]'
    roi = img_bgr[65:100, 140:200]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Binary threshold isolates the text from banners/flicker
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    return thresh

def run_v97_frame_perfect_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. FORCE F1
    floor_library = [{"floor": 1, "idx": 0}]
    img_f1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), img_f1)
    
    last_stage_state = get_stage_fingerprint(img_f1)
    
    print(f"--- Running v9.7 Frame-Perfect Auditor ---")

    for i in range(1, len(buffer_files)):
        if i % 1000 == 0: 
            print(f" [Scanning] Frame {i} | Floors Found: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. ANCHOR CHECK (Teleport detection)
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            # Is player at a verified spawn anchor?
            if any(abs(current_x - a) <= 3 for a in VALID_ANCHORS):
                
                # 3. ZERO-BUFFER STATE GATE
                # Instead of time, we check if the Stage ID has physically changed shape
                current_stage_state = get_stage_fingerprint(img_bgr)
                
                # Check for pixel difference in the binarized text area
                diff = cv2.absdiff(current_stage_state, last_stage_state)
                if np.sum(diff) > 500: # Significant text change detected
                    floor_num = len(floor_library) + 1
                    last_stage_state = current_stage_state.copy()
                    
                    bgr_prev = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
                    # Forensic overlays
                    cv2.putText(bgr_prev, f"F{i-1} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                    cv2.putText(img_bgr, f"F{i} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), 
                                np.hstack((bgr_prev, img_bgr)))
                    
                    floor_library.append({"floor": floor_num, "idx": i})
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i} | Rapid Shift Detected")

    with open(f"Run_0_FloorMap_v97.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v97_frame_perfect_audit()