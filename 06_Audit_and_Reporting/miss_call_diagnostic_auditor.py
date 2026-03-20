import cv2
import numpy as np
import os

# --- PRODUCTION CONSTANTS (MATCHING v18.6) ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]

def run_v19_2_robust_diagnostic(target_idx):
    buffer_root = "capture_buffer_0"
    out_dir = "diagnostic_results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Resolve Filename from Sorted Index
    if not os.path.exists(buffer_root):
        print(f"Error: Buffer root '{buffer_root}' not found.")
        return

    all_files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if target_idx >= len(all_files):
        print(f"Error: Index {target_idx} out of range (Total files: {len(all_files)})")
        return

    target_filename = all_files[target_idx]
    img_path = os.path.join(buffer_root, target_filename)
    img_bgr = cv2.imread(img_path)
    
    if img_bgr is None:
        print(f"Error: Could not load {img_path}")
        return
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Robust Template Loading
    ore_tpls = {}
    valid_states = ['act', 'sha'] # The only drawers we have
    
    if not os.path.exists("templates"):
        print("Error: 'templates' folder not found.")
        return

    for f in os.listdir("templates"):
        # Expecting format: tier_state.png (e.g., dirt1_act.png)
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            parts = f.split("_")
            tier = parts[0]
            state = parts[1].replace(".png", "")
            
            # THE FIX: Only process if the state is act or sha
            if state in valid_states:
                tpl_path = os.path.join("templates", f)
                raw_tpl = cv2.imread(tpl_path, 0)
                if raw_tpl is not None:
                    tpl = cv2.resize(raw_tpl, (48, 48))
                    if tier not in ore_tpls:
                        ore_tpls[tier] = {'act': [], 'sha': []}
                    ore_tpls[tier][state].append(tpl)
            else:
                # Skip things like ui_icon.png or progress_bar.png
                continue

    print(f"--- Diagnosing Sorted Index: {target_idx} ---")
    print(f"--- Resolved Filename: {target_filename} ---")
    
    # 3. Analyze Leftmost Slots (0, 1, 2)
    for slot_idx in range(3):
        cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
        cy = SLOT1_CENTER[1]
        
        # ROI Extraction (The Sliver)
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]
        
        # Draw Visual Markers
        cv2.rectangle(img_bgr, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
        cv2.putText(img_bgr, f"Slot {slot_idx}", (cx-20, cy-30), 0, 0.4, (0, 255, 0), 1)

        print(f"\n[Slot {slot_idx}] Match Scores:")
        
        results = []
        for tier, states in ore_tpls.items():
            for s_key in valid_states:
                for tpl in states[s_key]:
                    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, _ = cv2.minMaxLoc(res)
                    results.append((tier, s_key, max_v))
        
        # Sort and Display Top 5
        results.sort(key=lambda x: x[2], reverse=True)
        for tier, state, score in results[:5]:
            status = "PASS" if score > 0.77 else "FAIL"
            print(f"  > {tier}_{state}: {score:.4f} ({status})")

    # 4. Save Output
    diag_out_name = f"diag_v19_2_Idx{target_idx:05}_{target_filename}"
    cv2.imwrite(os.path.join(out_dir, diag_out_name), img_bgr)
    print(f"\n[FINISH] Diagnostic map saved to: {out_dir}/{diag_out_name}")

if __name__ == "__main__":
    run_v19_2_robust_diagnostic(target_idx=2049)