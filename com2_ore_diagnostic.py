import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_FLOORS = [6, 18, 21, 26, 30, 43, 67, 80]
BUFFER_ROOT = "capture_buffer_0"
UNIFIED_ROOT = "Unified_Consensus_Inputs/Run_0"

def run_consensus_recovery():
    # 1. Group Templates by Class Class
    # Classes: dirt, com, rare, epic, leg, myth, div
    class_groups = {}
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        name_parts = f.split("_")[0] # e.g. 'com2'
        cls = ''.join([i for i in name_parts if not i.isdigit()]) # 'com'
        if cls not in class_groups: class_groups[cls] = []
        
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            class_groups[cls].append({'tier': name_parts, 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    print(f"--- Step 3: Class Consensus & Temporal Recovery ---")

    for f_num in TARGET_FLOORS:
        f_idx = seq[f_num]['idx']
        raw_img = cv2.imread(os.path.join(UNIFIED_ROOT, seq[f_num]['frame']), 0)
        
        # --- PHASE A: FLOOR CLASS ELECTION ---
        # For each class, find which tier dominates this floor's high-conf slots
        floor_identities = {} # { 'com': 'com2', 'rare': 'rare1' ... }
        
        for cls, templates in class_groups.items():
            tier_votes = {} # { 'com1': cumulative_score }
            for slot in range(24):
                row, col = divmod(slot, 6)
                x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
                roi = raw_img[y1:y1+48, x1:x1+48]
                
                for t in templates:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED)
                    score = res.max()
                    if score > 0.85: # Only count strong signal for the 'election'
                        tier_votes[t['tier']] = tier_votes.get(t['tier'], 0) + score
            
            if tier_votes:
                floor_identities[cls] = max(tier_votes, key=tier_votes.get)

        print(f"\n[Floor {f_num}] Consensus Identities: {list(floor_identities.values())}")

        # --- PHASE B: TEMPORAL RECOVERY (For Overlaps) ---
        target_slots = [2] if f_num != 80 else [13]
        for slot_id in target_slots:
            row, col = divmod(slot_id, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            
            best_temp_score = 0
            best_temp_label = "None"

            # Search +/- 20 frames for a 'clean' version of the consensus tiers
            for off in range(-20, 21):
                idx = f_idx + off
                if 0 <= idx < len(buffer_files):
                    f_roi = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[idx]), 0)[y1:y1+48, x1:x1+48]
                    if np.max(f_roi) > 245: continue # Skip if Fairy/Crosshair is still there
                    
                    # Only match against the TIERS that won the election for this floor
                    for cls, winning_tier in floor_identities.items():
                        # Find the template matching the winning tier
                        valid_t = [t for t in class_groups[cls] if t['tier'] == winning_tier]
                        for t in valid_t:
                            res = cv2.matchTemplate(f_roi, t['img'], cv2.TM_CCORR_NORMED)
                            if res.max() > best_temp_score:
                                best_temp_score = res.max()
                                best_temp_label = t['tier']
            
            print(f"  > Slot {slot_id} Recovery: {best_temp_label} ({best_temp_score:.3f})")

if __name__ == "__main__":
    run_consensus_recovery()