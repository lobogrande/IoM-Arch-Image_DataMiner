import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v91_Scrubber"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.92

def clean_frame(current_f, history_stack):
    """Uses a 5-frame median to erase fairies and damage numbers."""
    if len(history_stack) < 5: return current_f
    # Temporal Median Eraser
    median = np.median(np.stack(history_stack, axis=0), axis=0).astype(np.uint8)
    return median

def run_v91_scrubber_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    history = []
    last_trigger_idx = 0
    
    print(f"--- Running v9.1 Forensic Scrubber ---")

    for i in range(len(buffer_files)):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        frame_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if frame_bgr is None: continue
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Maintain 5-frame history for the Median Eraser
        history.append(frame_gray)
        if len(history) > 5: history.pop(0)
        
        if len(history) < 5: continue

        # 1. GENERATE CLEANED BACKGROUND
        clean_img = clean_frame(frame_gray, history)

        # 2. TRIGGER LOGIC: Look for the Player "Snap"
        # We search the original frame so we don't 'median-out' the player
        search_roi = frame_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            anchor = next((a for a in VALID_ANCHORS if abs(current_x - a) <= 4), None)
            
            if anchor is not None:
                # 3. VERIFICATION: Did the CLEANED grid actually change?
                # We compare current clean_img vs. the last clean_img
                if i > 5:
                    prev_clean = clean_frame(history[0], history) # Oldest in stack
                    diff = cv2.absdiff(clean_img[200:450, :], prev_clean[200:450, :])
                    # If total difference is high, it's a real floor reset
                    if np.sum(diff > 50) > 3000 and (i - last_trigger_idx) > 15:
                        floor_num = len(floor_library) + 1
                        last_trigger_idx = i
                        
                        # Save the audit JPG
                        bgr_end = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-5]))
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), 
                                    np.hstack((bgr_end, frame_bgr)))
                        
                        floor_library.append({"floor": floor_num, "idx": i})
                        print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i} | Anchor: {anchor}")

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v91_scrubber_audit()