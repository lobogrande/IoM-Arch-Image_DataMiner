import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
# Floors with known transient noise (Fairies, Players, Crosshairs)
TARGET_FLOORS = [6, 18, 21, 26, 30, 43, 67, 80]
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/Step3_VisualForensics_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_visual_forensics():
    # 1. Load Templates (Consensus-Ready)
    class_groups = {}
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]) or not f.endswith('.png'): continue
        name = f.split("_")[0]
        cls = ''.join([i for i in name if not i.isdigit()])
        if cls not in class_groups: class_groups[cls] = []
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            class_groups[cls].append({'tier': name, 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}
    
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    print(f"--- Running Step 3 Visual Forensics ---")

    for f_num in TARGET_FLOORS:
        if f_num not in seq: continue
        f_idx = seq[f_num]['idx']
        
        # Load the original "Main" frame
        raw_bgr = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{seq[f_num]['frame']}"))
        if raw_bgr is None: raw_bgr = cv2.imread(os.path.join(UNIFIED_ROOT, seq[f_num]['frame']))
        if raw_bgr is None: continue
        
        # --- Consensus Phase ---
        floor_identities = {}
        gray_main = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        for cls, templates in class_groups.items():
            tier_votes = {}
            for slot in range(24):
                row, col = divmod(slot, 6)
                x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
                roi = gray_main[y1:y1+48, x1:x1+48]
                for t in templates:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED)
                    if res.max() > 0.88:
                        tier_votes[t['tier']] = tier_votes.get(t['tier'], 0) + res.max()
            if tier_votes: floor_identities[cls] = max(tier_votes, key=tier_votes.get)

        # --- Target Recovery Phase ---
        # Identifying the slot likely to have the Fairy/Player
        target_slot = 2 if f_num != 80 else 13
        row, col = divmod(target_slot, 6)
        tx1, ty1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        
        # Mark the suspect slot in YELLOW (Transient Alert) on the original
        cv2.rectangle(raw_bgr, (tx1, ty1), (tx1+48, ty1+48), (0, 255, 255), 2)
        cv2.putText(raw_bgr, "SUSPECT", (tx1, ty1-5), 0, 0.4, (0, 255, 255), 1)

        # Buffer Surf
        best_s, best_l, best_idx = 0, "None", -1
        for off in range(-35, 36):
            idx = f_idx + off
            if 0 <= idx < len(buffer_files):
                f_path = os.path.join(BUFFER_ROOT, buffer_files[idx])
                f_img = cv2.imread(f_path, 0)
                if f_img is None: continue
                f_roi = f_img[ty1:ty1+48, tx1:tx1+48]
                
                # Exclusion: skip if the transient noise is still in this frame
                if np.max(f_roi) > 242: continue 
                
                for cls, win in floor_identities.items():
                    v_t = [t for t in class_groups[cls] if t['tier'] == win]
                    for t in v_t:
                        res = cv2.matchTemplate(f_roi, t['img'], cv2.TM_CCORR_NORMED)
                        if res.max() > best_s:
                            best_s, best_l, best_idx = res.max(), t['tier'], idx

        # --- Visual Export ---
        if best_idx != -1:
            clean_frame = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[best_idx]))
            # Annotate the "Truth" frame
            cv2.rectangle(clean_frame, (tx1, ty1), (tx1+48, ty1+48), (0, 255, 0), 2)
            cv2.putText(clean_frame, f"RECOVERED: {best_l} ({best_s:.3f})", 
                        (tx1-20, ty1+60), 0, 0.4, (0, 255, 0), 1)
            cv2.putText(clean_frame, f"Frame: {buffer_files[best_idx]}", 
                        (30, 30), 0, 0.6, (0, 255, 0), 2)

            # Create side-by-side comparison
            # Ensuring both images are same height for hstack
            h1, w1 = raw_bgr.shape[:2]
            h2, w2 = clean_frame.shape[:2]
            comparison = np.hstack((raw_bgr, clean_frame))
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"F{f_num}_Recovery_Proof.jpg"), comparison)
            print(f" [+] Exported comparison for Floor {f_num}")

if __name__ == "__main__":
    run_visual_forensics()