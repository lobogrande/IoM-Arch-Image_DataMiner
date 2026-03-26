import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
# Surgical targets: Slots identified previously as Purple/Cyan/Obstructed
MANUAL_TARGETS = {
    6:  [16, 19], # Crosshairs
    18: [0],      # Player (should reveal blank)
    30: [1, 15],  # Player and Crosshair
    43: [1, 7],   # Player and Block
    67: [2],      # Fairy/Crosshair
    80: [12, 13]  # Crosshair and Fairy
}

BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/Surgical_v62_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_occupancy_dna(img_gray, bg_templates):
    """Generates a 24-bit DNA map. Scan stops if this changes."""
    dna = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        # Standard delta check for occupancy
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        dna.append(diff > 6.0) 
    return dna

def run_v62_reconstruction():
    # 1. Asset Loading
    bg_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    all_block_t = []
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if any(x in f for x in ["background", "negative"]) or not f.endswith('.png'): continue
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is not None: all_block_t.append({'tier': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    # 2. Sequence Mapping
    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    print(f"--- Running v6.2 Bidirectional Forensic Reconstruction ---")

    for f_num, target_slots in MANUAL_TARGETS.items():
        if f_num not in seq: continue
        
        # FIX: Pull the anchor frame directly from the BUFFER, not Unified
        anchor_name = seq[f_num]['frame']
        anchor_idx = seq[f_num]['idx']
        anchor_path = os.path.join(BUFFER_ROOT, anchor_name)
        
        anchor_bgr = cv2.imread(anchor_path)
        if anchor_bgr is None:
            print(f" [!] Error: Could not find anchor frame {anchor_name} in {BUFFER_ROOT}")
            continue
            
        anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)
        floor_dna = get_occupancy_dna(anchor_gray, bg_t)
        
        # Prepare the reconstructed frame
        recon_bgr = anchor_bgr.copy()
        
        for slot_id in target_slots:
            row, col = divmod(slot_id, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            
            # Highlight SUSPECT on original
            cv2.rectangle(anchor_bgr, (x1, y1), (x1+48, y1+48), (0, 255, 255), 2)
            
            # Bidirectional Search
            best_results = {'back': None, 'forward': None}
            
            for direction, step in [('back', -1), ('forward', 1)]:
                # Scan up to 200 frames out, but DNA guardrail will likely stop us sooner
                for off in range(1, 201):
                    idx = anchor_idx + (off * step)
                    if not (0 <= idx < len(buffer_files)): break
                    
                    check_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[idx]), 0)
                    if check_img is None: continue
                    
                    # GUARDRAIL: If the floor layout changes, the scan for this direction is DEAD.
                    if get_occupancy_dna(check_img, bg_t) != floor_dna: break
                    
                    roi = check_img[y1:y1+48, x1:x1+48]
                    # Check for transient clearance (Fairy/Crosshair intensity check)
                    if np.max(roi) < 240:
                        # Success: Identify what is in this clean spot
                        best_o, best_l = 0, "BLANK"
                        for t in all_block_t:
                            res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED)
                            if res.max() > best_o: best_o, best_l = res.max(), t['tier']
                        
                        best_results[direction] = (best_l, best_o, idx)
                        break

            # Consolidate Results
            if best_results['back'] and best_results['forward']:
                b_label, _, b_idx = best_results['back']
                f_label, _, _ = best_results['forward']
                
                # Apply result if consensus is met, otherwise flag conflict
                final_label = b_label if b_label == f_label else f"CONFLICT({b_label}/{f_label})"
                
                # Patch the 'best' crop (backward choice by default) into the recon image
                clean_patch = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[b_idx]))
                recon_bgr[y1:y1+48, x1:x1+48] = clean_patch[y1:y1+48, x1:x1+48]
                
                cv2.rectangle(recon_bgr, (x1, y1), (x1+48, y1+48), (0, 255, 0), 2)
                cv2.putText(recon_bgr, f"{final_label}", (x1-10, y1+60), 0, 0.4, (0, 255, 0), 1)

        comparison = np.hstack((anchor_bgr, recon_bgr))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"F{f_num}_Forensic_Consensus.jpg"), comparison)
        print(f" [+] Exported Forensic Consensus for Floor {f_num}")

if __name__ == "__main__":
    run_v41_reconstruction = run_v62_reconstruction # Maintaining logic for internal call
    run_v62_reconstruction()