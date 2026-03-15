import cv2
import numpy as np
import os

# --- PRODUCTION CONSTANTS (MUST MATCH v18.6) ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]

def run_v19_1_positional_diagnostic(target_idx):
    buffer_root = "capture_buffer_0"
    out_dir = "diagnostic_results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Resolve the Filename from the Sorted Index
    # This mirrors exactly how the production script sees the data
    all_files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if target_idx >= len(all_files):
        print(f"Error: Index {target_idx} is out of range. Buffer only has {len(all_files)} files.")
        return

    # THE RESOLUTION STEP
    target_filename = all_files[target_idx]
    img_path = os.path.join(buffer_root, target_filename)
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not load {img_path}")
        return
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Load Templates
    ore_tpls = {}
    if not os.path.exists("templates"):
        print("Error: 'templates' folder not found.")
        return

    for f in os.listdir("templates"):
        if "_" in f and f.endswith(".png") and not f.startswith("background"):
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            tpl_path = os.path.join("templates", f)
            raw_tpl = cv2.imread(tpl_path, 0)
            if raw_tpl is not None:
                tpl = cv2.resize(raw_tpl, (48, 48))
                if tier not in ore_tpls: ore_tpls[tier] = {'act': [], 'sha': []}
                ore_tpls[tier][state].append(tpl)

    print(f"--- Diagnosing Sorted Index: {target_idx} ---")
    print(f"--- Resolved Filename: {target_filename} ---")
    
    # 3. Analyze the Leftmost Slots (0, 1, 2)
    for slot_idx in range(3):
        cx = int(SLOT1_CENTER[0] + (slot_idx * STEP_X))
        cy = SLOT1_CENTER[1]
        
        # ROI Extraction
        roi = img_gray[cy-24:cy+24, cx-24:cx+24]
        
        # Draw Visual Markers on Output
        cv2.rectangle(img_bgr, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
        cv2.putText(img_bgr, f"Slot {slot_idx}", (cx-20, cy-30), 0, 0.4, (0, 255, 0), 1)

        print(f"\n[Slot {slot_idx}] Match Scores:")
        
        results = []
        for tier, states in ore_tpls.items():
            for state_key in ['act', 'sha']:
                for tpl in states[state_key]:
                    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, _ = cv2.minMaxLoc(res)
                    results.append((tier, state_key, max_v))
        
        # Sort by match confidence
        results.sort(key=lambda x: x[2], reverse=True)
        
        for tier, state, score in results[:5]:
            status = "PASS" if score > 0.77 else "FAIL"
            print(f"  > {tier}_{state}: {score:.4f} ({status})")

    # Save Diagnostic Image
    diag_out_name = f"diag_Idx{target_idx:05}_{target_filename}"
    cv2.imwrite(os.path.join(out_dir, diag_out_name), img_bgr)
    print(f"\n[FINISH] Diagnostic results saved to {out_dir}/{diag_out_name}")

if __name__ == "__main__":
    # We are probing Index 2049, which corresponds to your F28 mis-call
    run_v19_1_positional_diagnostic(target_idx=2049)