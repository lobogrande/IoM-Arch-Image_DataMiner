import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = [6, 18, 21, 26, 30, 43, 67, 80]
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"

def run_consensus_recovery():
    # 1. Group Templates by Class (dirt, com, rare, epic, leg, myth, div)
    class_groups = {}
    if not os.path.exists("templates"):
        print("[!] Error: 'templates' folder not found.")
        return

    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]) or not f.endswith('.png'):
            continue
        name_parts = f.split("_")[0] 
        cls = ''.join([i for i in name_parts if not i.isdigit()]) 
        if cls not in class_groups: class_groups[cls] = []
        
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            class_groups[cls].append({'tier': name_parts, 'img': cv2.resize(img, (48, 48))})

    # 2. Load Sequence Data
    seq_path = os.path.join(UNIFIED_ROOT, "final_sequence.json")
    if not os.path.exists(seq_path):
        print(f"[!] Error: Sequence file not found at {seq_path}")
        return
        
    with open(seq_path, 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}
    
    if not os.path.exists(BUFFER_ROOT):
        print(f"[!] Error: Buffer root {BUFFER_ROOT} not found.")
        return
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    print(f"--- Step 3 Diagnostic (v5.7): Consensus & Temporal Recovery ---")

    for f_num in TARGET_FLOORS:
        if f_num not in seq:
            print(f" [?] Floor {f_num} not in sequence. Skipping.")
            continue
            
        f_name = seq[f_num]['frame']
        f_idx = seq[f_num]['idx']
        
        # Try both common naming conventions
        possible_paths = [
            os.path.join(UNIFIED_ROOT, f_name),
            os.path.join(UNIFIED_ROOT, f"F{f_num}_{f_name}")
        ]
        
        raw_img = None
        for p in possible_paths:
            if os.path.exists(p):
                raw_img = cv2.imread(p, 0)
                break
        
        if raw_img is None:
            print(f" [!] Error: Could not load image for Floor {f_num} (Searched: {possible_paths})")
            continue
        
        # --- PHASE A: FLOOR CLASS ELECTION ---
        floor_identities = {} 
        for cls, templates in class_groups.items():
            tier_votes = {} 
            for slot in range(24):
                row, col = divmod(slot, 6)
                x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
                # Safety check for ROI bounds
                if y1 < 0 or x1 < 0 or y1+48 > raw_img.shape[0] or x1+48 > raw_img.shape[1]: continue
                
                roi = raw_img[y1:y1+48, x1:x1+48]
                for t in templates:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED)
                    score = res.max()
                    if score > 0.85:
                        tier_votes[t['tier']] = tier_votes.get(t['tier'], 0) + score
            
            if tier_votes:
                floor_identities[cls] = max(tier_votes, key=tier_votes.get)

        print(f"\n[Floor {f_num}] Consensus: {list(floor_identities.values())}")

        # --- PHASE B: TEMPORAL RECOVERY ---
        target_slots = [2] if f_num != 80 else [13]
        for slot_id in target_slots:
            row, col = divmod(slot_id, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            
            best_temp_score = 0
            best_temp_label = "None"

            for off in range(-30, 31): # Expanded search window to +/- 30 frames
                idx = f_idx + off
                if 0 <= idx < len(buffer_files):
                    f_path = os.path.join(BUFFER_ROOT, buffer_files[idx])
                    f_img = cv2.imread(f_path, 0)
                    if f_img is None: continue
                    
                    f_roi = f_img[y1:y1+48, x1:x1+48]
                    # Skip frame if it's currently obscured by high-intensity noise
                    if np.max(f_roi) > 242: continue 
                    
                    for cls, winning_tier in floor_identities.items():
                        valid_t = [t for t in class_groups[cls] if t['tier'] == winning_tier]
                        for t in valid_t:
                            res = cv2.matchTemplate(f_roi, t['img'], cv2.TM_CCORR_NORMED)
                            if res.max() > best_temp_score:
                                best_temp_score = res.max()
                                best_temp_label = t['tier']
            
            print(f"  > Slot {slot_id} Recovery Result: {best_temp_label} ({best_temp_score:.3f})")

if __name__ == "__main__":
    run_consensus_recovery()