import cv2
import numpy as np
import os
import json

# --- TARGET CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 2 # Or 7
# Choose a "Ghost" slot and a "Missed" slot from your HUD images
TARGET_SLOTS = [1, 2, 6, 9, 10, 11, 13, 14, 19] 

SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
UNIFIED_ROOT = "Unified_Consensus_Inputs"

def run_forensic_inspector():
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)

    # 1. Load Templates
    ore_templates = []
    bg_templates = []
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"):
            bg_templates.append({'name': f, 'img': img})
        else:
            ore_templates.append({'name': f, 'img': img})

    # 2. Load Floor Image
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    files = [f for f in os.listdir(run_path) if f.startswith(f"F{TARGET_FLOOR}_")]
    raw_img = cv2.imread(os.path.join(run_path, files[0]))
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    print(f"--- Forensic Audit: Run {TARGET_RUN} Floor {TARGET_FLOOR} ---")

    for slot in TARGET_SLOTS:
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        roi = gray[cy-24:cy+24, cx-24:cx+24]
        
        # Test Ore Matches (Masked)
        ore_results = []
        for t in ore_templates:
            res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
            _, score, _, _ = cv2.minMaxLoc(res)
            ore_results.append((t['name'], score, t['img']))
        ore_results.sort(key=lambda x: x[1], reverse=True)

        # Test BG Matches (Standard - No Mask)
        bg_results = []
        for t in bg_templates:
            score = cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED).max()
            bg_results.append((t['name'], score, t['img']))
        bg_results.sort(key=lambda x: x[1], reverse=True)

        # Create Comparison Strip
        best_ore_name, best_ore_score, best_ore_img = ore_results[0]
        best_bg_name, best_bg_score, best_bg_img = bg_results[0]

        # Annotate images
        roi_color = raw_img[cy-24:cy+24, cx-24:cx+24].copy()
        ore_vis = cv2.cvtColor(best_ore_img, cv2.COLOR_GRAY2BGR)
        bg_vis = cv2.cvtColor(best_bg_img, cv2.COLOR_GRAY2BGR)

        cv2.putText(roi_color, "RAW", (2, 10), 0, 0.3, (0,255,255), 1)
        cv2.putText(ore_vis, f"ORE:{best_ore_score:.2f}", (2, 10), 0, 0.3, (0,255,0), 1)
        cv2.putText(bg_vis, f"BG:{best_bg_score:.2f}", (2, 10), 0, 0.3, (255,255,0), 1)

        strip = np.hstack((roi_color, ore_vis, bg_vis))
        cv2.imwrite(f"Forensic_F{TARGET_FLOOR}_Slot{slot}.png", strip)
        
        print(f"\n[Slot {slot}]")
        print(f"  Best Ore: {best_ore_score:.4f} ({best_ore_name})")
        print(f"  Best BG : {best_bg_score:.4f} ({best_bg_name})")
        print(f"  Delta   : {best_ore_score - (1.0 - best_bg_score):.4f}")

if __name__ == "__main__":
    run_forensic_inspector()