import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- MATCH PRODUCTION v23.0 ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]

def run_v24_path_trace(target_idx):
    buffer_root = cfg.get_buffer_path(0)
    out_dir = "diagnostic_v24"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Resolve File
    all_files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    target_file = all_files[target_idx]
    img_bgr = cv2.imread(os.path.join(buffer_root, target_file))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Load Templates
    ore_tpls = {'ore': {}}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            img = cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48))
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: ore_tpls['ore'][tier][state].append(img)

    print(f"--- Diagnosing Path Audit for Index {target_idx} ({target_file}) ---")

    # 3. Simulate Player Detection
    p_right = cv2.imread("templates/player_right.png", 0)
    res = cv2.matchTemplate(img_gray[150:300, 0:480], p_right, cv2.TM_CCOEFF_NORMED)
    _, max_v, _, max_loc = cv2.minMaxLoc(res)
    n_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
    
    print(f"  [Spatial] Player Slot: {n_slot} | Max Match: {max_v:.4f}")

    if n_slot is None: return

    # 4. TRACE THE PATH AUDIT
    print("\n--- BEGIN PATH TRACE ---")
    
    # Check N-1 Sliver
    if n_slot > 0:
        s_idx = n_slot - 1
        cx, cy = int(SLOT1_CENTER[0] + (s_idx * STEP_X)), SLOT1_CENTER[1]
        roi_sliver = img_gray[cy-24:cy+24, cx-24:cx-12] # Left 12px
        
        best_sha = 0
        for tier in ore_tpls['ore']:
            for t_img in ore_tpls['ore'][tier]['sha']:
                score = cv2.matchTemplate(roi_sliver, t_img[:, :12], cv2.TM_CCOEFF_NORMED).max()
                best_sha = max(best_sha, score)
        
        status = "DIRTY (Correctly Discarded)" if best_sha > 0.77 else "CLEAN (Potential Miss-call!)"
        print(f"  [Slot {s_idx} Sliver] Best Shadow Match: {best_sha:.4f} -> Result: {status}")
        
        # Save Crop for Visual Confirmation
        cv2.imwrite(f"{out_dir}/Idx{target_idx}_Slot{s_idx}_Sliver.png", roi_sliver)

    # Check N-2 to 0 Full Tiles
    if n_slot >= 2:
        for s_idx in range(n_slot - 1):
            cx, cy = int(SLOT1_CENTER[0] + (s_idx * STEP_X)), SLOT1_CENTER[1]
            roi_full = img_gray[cy-24:cy+24, cx-24:cx+24]
            
            best_ore = 0
            for tier in ore_tpls['ore']:
                for state in ['act', 'sha']:
                    for t_img in ore_tpls['ore'][tier][state]:
                        score = cv2.matchTemplate(roi_full, t_img, cv2.TM_CCOEFF_NORMED).max()
                        best_ore = max(best_ore, score)

            status = "DIRTY" if best_ore > 0.77 else "CLEAN"
            print(f"  [Slot {s_idx} Full] Best Ore/Sha Match: {best_ore:.4f} -> Result: {status}")
            cv2.imwrite(f"{out_dir}/Idx{target_idx}_Slot{s_idx}_FullTile.png", roi_full)

    print("\n[FINISH] Diagnostic images saved to 'diagnostic_v24/'")

if __name__ == "__main__":
    # Provide the Index of the Floor 4 duplicate here:
    run_v24_path_trace(target_idx=75)