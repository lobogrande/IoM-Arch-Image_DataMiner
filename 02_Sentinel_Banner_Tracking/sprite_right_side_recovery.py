# sprite_right_side_recovery.py
# Purpose: Recover "missing" detections in Slots 4 & 5 using bottom-half matching.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS COORDINATES (AI TOP-LEFT)
S0_X, S0_Y = 11, 225
STEP = 59.0
TPL_W, TPL_H = 40, 60
LIMIT = 5000  # Limited set for rapid proof

def run_recovery_proof():
    print(f"--- RIGHT-SIDE RECOVERY PROOF (S4 & S5 | 0-{LIMIT}) ---")
    
    # 1. Prepare Bottom-Half Template
    full_tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    bottom_tpl = full_tpl[30:, :] # Use bottom 30px (feet/shadow)
    bh, bw = bottom_tpl.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:LIMIT]
    
    # Target Coordinates
    targets = {
        4: int(S0_X + (4 * STEP)),
        5: int(S0_X + (5 * STEP))
    }

    hits = []
    if not os.path.exists("recovery_debug"): os.makedirs("recovery_debug")

    for f_idx, filename in enumerate(files):
        img_gray = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img_gray is None: continue

        for slot_id, x_tl in targets.items():
            # Extract the ROI where the FEET should be
            # AI Top-Left Y is 225, so Feet start at 225 + 30 = 255
            roi = img_gray[255 : 255+bh, x_tl : x_tl+bw]
            
            if roi.shape[0] < bh: continue
            
            res = cv2.matchTemplate(roi, bottom_tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)

            # We found a pulse! (Threshold 0.78 based on your pulse audit)
            if val > 0.78:
                hits.append({'frame': f_idx, 'slot': slot_id, 'conf': round(val, 4)})
                
                # SAVE VISUAL PROOF for the first 5 hits per slot
                if len([h for h in hits if h['slot'] == slot_id]) <= 5:
                    vis = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename))
                    # Draw HUD box (40x60 centered on feet)
                    cv2.rectangle(vis, (x_tl, 225), (x_tl+40, 225+60), (0, 255, 255), 1)
                    cv2.putText(vis, f"RECOVERED S{slot_id} | Conf: {round(val,2)}", 
                                (x_tl-20, 220), 0, 0.4, (0, 255, 255), 1)
                    cv2.imwrite(f"recovery_debug/recovered_s{slot_id}_f{f_idx}.jpg", vis)

        if f_idx % 1000 == 0:
            print(f"  Scanning... {f_idx}/{LIMIT} | Hits: {len(hits)}")

    print(f"\n[DONE] Recovered {len(hits)} hits in Slots 4 & 5.")
    print("Check 'recovery_debug/' to see the visual proof.")

if __name__ == "__main__":
    run_recovery_proof()