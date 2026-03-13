import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
# Hard-coded for this validation pass as requested
MANUAL_TARGETS = {
    6:  [16, 19], # Crosshairs
    18: [0],      # Player (should reveal blank)
    30: [1, 15],  # Player and Crosshair
    43: [1, 7],   # Player and Ore
    67: [2],      # Fairy/Crosshair
    80: [12, 13]  # Crosshair and Fairy
}

BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/Surgical_Reconstruction_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_occupancy_dna(img_gray, bg_templates):
    """Creates a boolean map of the floor layout."""
    dna = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        dna.append(diff > 6.0) # True if occupied
    return dna

def run_surgical_reconstruction():
    # 1. Assets
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    all_ore_t = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]) or not f.endswith('.png'): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None: all_ore_t.append({'tier': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    print(f"--- Running v6.1 Bidirectional Surgical Reconstruction ---")

    for f_num, target_slots in MANUAL_TARGETS.items():
        if f_num not in seq: continue
        f_idx = seq[f_num]['idx']
        
        # Load anchor frame
        anchor_bgr = cv2.imread(os.path.join(UNIFIED_ROOT, seq[f_num]['frame']))
        if anchor_bgr is None: anchor_bgr = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{seq[f_num]['frame']}"))
        anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)
        
        # Lock the Floor DNA
        floor_dna = get_occupancy_dna(anchor_gray, bg_t)
        
        recovered_data = {} # {slot_id: {'label': ID, 'b_idx': idx, 'f_idx': idx}}

        for slot_id in target_slots:
            row, col = divmod(slot_id, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            
            direction_hits = {'back': None, 'forward': None}
            
            for direction, step in [('back', -1), ('forward', 1)]:
                for off in range(1, 150): # Deep search
                    idx = f_idx + (off * step)
                    if not (0 <= idx < len(buffer_files)): break
                    
                    f_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[idx]), 0)
                    # Guardrail: Ensure floor layout hasn't changed
                    if get_occupancy_dna(f_img, bg_t) != floor_dna: break
                    
                    roi = f_img[y1:y1+48, x1:x1+48]
                    # Check for "Clearance" (No high-intensity Fairy/X-hair signal)
                    if np.max(roi) < 240:
                        # Identify
                        best_o, best_l = 0, "BLANK"
                        for t in all_ore_t:
                            res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED)
                            if res.max() > best_o: best_o, best_l = res.max(), t['tier']
                        
                        direction_hits[direction] = (best_l, best_o, idx)
                        break
            
            # Cross-Validation Logic
            if direction_hits['back'] and direction_hits['forward']:
                b_label, b_score, b_idx = direction_hits['back']
                f_label, f_score, f_idx_hit = direction_hits['forward']
                
                if b_label == f_label: # Consensus reached
                    recovered_data[slot_id] = {'label': b_label, 'score': b_score, 'idx': b_idx}
                else:
                    recovered_data[slot_id] = {'label': f"CONFLICT({b_label}/{f_label})", 'score': 0, 'idx': b_idx}

        # --- Visual Synthesis ---
        # Draw Original with Suspects
        left_panel = anchor_bgr.copy()
        right_panel = anchor_bgr.copy() # We will patch in recovered crops

        for slot_id in target_slots:
            row, col = divmod(slot_id, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            cv2.rectangle(left_panel, (x1, y1), (x1+48, y1+48), (0, 255, 255), 2)
            
            if slot_id in recovered_data:
                res = recovered_data[slot_id]
                # Grab the clean crop from the buffer
                clean_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[res['idx']]))
                right_panel[y1:y1+48, x1:x1+48] = clean_img[y1:y1+48, x1:x1+48]
                
                cv2.rectangle(right_panel, (x1, y1), (x1+48, y1+48), (0, 255, 0), 2)
                cv2.putText(right_panel, f"{res['label']}", (x1-10, y1+60), 0, 0.4, (0, 255, 0), 1)

        comparison = np.hstack((left_panel, right_panel))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"F{f_num}_Surgical_Consensus.jpg"), comparison)
        print(f" [+] Verified Floor {f_num} via Bidirectional Consensus")

if __name__ == "__main__":
    run_surgical_reconstruction()