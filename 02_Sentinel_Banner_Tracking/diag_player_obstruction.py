# diag_player_obstruction.py
# Purpose: Find the "Invisible Player" - check why match scores are low on known obstructed slots.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# Targets where you expect "Likely Empty" but got "Dirt/Com"
TARGETS = [
    {'f': 2, 's': 1, 'name': 'R1_S1'},
    {'f': 3, 's': 3, 'name': 'R1_S3'},
    {'f': 7, 's': 0, 'name': 'R1_S0'}
]

ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE)

def run_autopsy():
    buffer_dir = cfg.get_buffer_path(0)
    boundaries = pd.read_csv(os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv"))
    player_tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "negative_player_0.png"), 0)
    player_tpl = cv2.resize(player_tpl, (SIDE_PX, SIDE_PX))
    
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.png')])
    results = []

    print(f"--- PLAYER OBSTRUCTION AUTOPSY ---")
    for target in TARGETS:
        row = boundaries[boundaries['floor_id'] == target['f']].iloc[0]
        f_idx = int(row['true_start_frame'])
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        col = target['s']
        cx = int(ORE0_X + (col * STEP))
        x1, y1 = int(cx - SIDE_PX//2), int(ORE0_Y - SIDE_PX//2)
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        
        # 1. Match Score
        res = cv2.matchTemplate(roi, player_tpl, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        
        # 2. Side-Slice Variance
        slot_48 = roi[4:52, 4:52]
        slice_roi = slot_48[:, 0:12]
        std_val = np.std(slice_roi)
        
        # 3. Complexity
        comp = cv2.Laplacian(roi, cv2.CV_64F).var()
        
        print(f"Floor {target['f']} {target['name']}: Match={score:.3f} | SliceStd={std_val:.2f} | Complexity={comp:.1f}")
        
        # Save crop for visual inspection
        cv2.imwrite(f"autopsy_f{target['f']}_{target['name']}.jpg", cv2.resize(roi, (200,200)))

if __name__ == "__main__":
    run_autopsy()