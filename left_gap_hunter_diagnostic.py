import cv2
import numpy as np
import os
import json

# --- CALIBRATED CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X = 59.1
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
SLIVER_WIDTH = 10  # Verified 'Sweet Spot' from v13.6
BG_MATCH_THRESHOLD = 0.92 
PLAYER_MATCH_THRESHOLD = 0.82 # Recovered from 0.85

def get_best_bg_match(roi, bg_tpls, is_sliver=False):
    best_score = 0
    # Use the 10px Sweet Spot to stay behind the player's back
    audit_roi = roi[:, :SLIVER_WIDTH] if is_sliver else roi
    
    for tpl in bg_tpls:
        audit_tpl = tpl[:, :SLIVER_WIDTH] if is_sliver else tpl
        if audit_roi.shape != audit_tpl.shape:
            audit_tpl = cv2.resize(audit_tpl, (audit_roi.shape[1], audit_roi.shape[0]))
            
        res = cv2.matchTemplate(audit_roi, audit_tpl, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        if score > best_score: best_score = score
    return best_score

def run_gap_recovery_v13_7():
    buffer_root = "capture_buffer_0"
    output_dir = "diagnostic_results/Gap_Recovery_v13_7"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Assets
    bg_tpls = []
    for f in os.listdir("templates"):
        if "background_plain_" in f:
            img = cv2.imread(os.path.join("templates", f), 0)
            if img is not None: bg_tpls.append(cv2.resize(img, (48, 48)))

    player_t = cv2.imread("templates/player_right.png", 0)
    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    print(f"--- Running v13.7 Calibrated Gap Recovery ---")
    
    found_count = 0
    for i in range(len(files)):
        if i % 1000 == 0: print(f" Scanning {i}...", end='\r')
        
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect Player with calibrated threshold
        res = cv2.matchTemplate(img_gray[150:400, 0:480], player_t, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        
        if max_v > PLAYER_MATCH_THRESHOLD:
            slot = next((idx for idx, a in enumerate(VALID_ANCHORS) if abs(max_loc[0] - a) <= 12), None)
            
            if slot is not None and slot > 0:
                all_clear = True
                
                # A. Distant slots (Full ROI)
                for s in range(slot - 1):
                    cx = int(SLOT1_CENTER[0] + (s * STEP_X))
                    roi = img_gray[261-24:261+24, cx-24:cx+24]
                    if get_best_bg_match(roi, bg_tpls, False) < BG_MATCH_THRESHOLD:
                        all_clear = False; break
                
                # B. Immediate-left slot (10px Sliver ROI)
                if all_clear:
                    cx_overlap = int(SLOT1_CENTER[0] + ((slot - 1) * STEP_X))
                    roi_overlap = img_gray[261-24:261+24, cx_overlap-24:cx_overlap+24]
                    if get_best_bg_match(roi_overlap, bg_tpls, True) < BG_MATCH_THRESHOLD:
                        all_clear = False
                
                if all_clear:
                    found_count += 1
                    cx_p = int(SLOT1_CENTER[0] + (slot * STEP_X))
                    discovery = img_bgr[261-45:261+45, 0:cx_p+40]
                    cv2.imwrite(os.path.join(output_dir, f"Gap_Slot{slot}_Idx{i:05}.png"), discovery)
                    print(f"\n [!] RECOVERED GAP: Frame {i} (Slot {slot}, Player Score {max_v:.2f})")

    print(f"\n[FINISH] {found_count} gaps recovered.")

if __name__ == "__main__":
    run_gap_recovery_v13_7()