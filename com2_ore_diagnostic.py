import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
# Hard-coded list for validation, but logic is now dynamic
TARGET_FLOORS = [6, 18, 30, 43, 67, 80]
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/Step3_V59_Forensics_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
INTENSITY_THRES = 242 # Fairy/Crosshair/UI signature
D_GATE = 6            # Occupancy delta

def get_floor_fingerprint(gray_img, bg_templates):
    """Creates a 24-bit map of which slots contain objects."""
    map = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = gray_img[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        map.append(diff > D_GATE)
    return map

def run_v59_visual_forensics():
    # 1. Load Assets & Sequence
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    player_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_player")]
    all_ore_t = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None: all_ore_t.append({'tier': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])

    for f_num in TARGET_FLOORS:
        if f_num not in seq: continue
        f_idx = seq[f_num]['idx']
        raw_bgr = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{seq[f_num]['frame']}"))
        if raw_bgr is None: continue
        gray_main = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        
        # A. Create Floor Fingerprint (To prevent drifting to other floors)
        origin_fingerprint = get_floor_fingerprint(gray_main, bg_t)
        
        # B. Identify ALL Suspect Slots (Transient overlaps)
        suspect_slots = []
        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray_main[y1:y1+48, x1:x1+48]
            
            # Check for Player, Fairy, or Crosshair
            best_p = max([cv2.matchTemplate(roi, pt, cv2.TM_CCORR_NORMED).max() for pt in player_t] + [0])
            if best_p > 0.85 or np.max(roi) > INTENSITY_THRES:
                suspect_slots.append(slot)
                # Mark Suspect on HUD
                cv2.rectangle(raw_bgr, (x1, y1), (x1+48, y1+48), (0, 255, 255), 2)
                label = "PLAYER" if best_p > 0.85 else "X-HAIR/FAIRY"
                cv2.putText(raw_bgr, label, (x1, y1-5), 0, 0.35, (0, 255, 255), 1)

        if not suspect_slots: continue

        # C. Temporal Surf with Guardrails
        clean_results = {}
        for slot_id in suspect_slots:
            row, col = divmod(slot_id, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            
            # Scan +/- 50 frames but BREAK if fingerprint changes
            for direction in [-1, 1]:
                for off in range(1, 51):
                    idx = f_idx + (off * direction)
                    if not (0 <= idx < len(buffer_files)): break
                    
                    # 1. Floor Boundary Guardrail
                    f_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[idx]), 0)
                    if f_img is None: continue
                    current_fingerprint = get_floor_fingerprint(f_img, bg_t)
                    if current_fingerprint != origin_fingerprint: break # STOP: We left the floor!

                    # 2. Check if Slot is now Clean
                    f_roi = f_img[y1:y1+48, x1:x1+48]
                    if np.max(f_roi) < INTENSITY_THRES:
                        # Success: Identify the ore in this clean frame
                        best_o, best_l = 0, "EMPTY"
                        for t in all_ore_t:
                            res = cv2.matchTemplate(f_roi, t['img'], cv2.TM_CCORR_NORMED)
                            if res.max() > best_o: best_o, best_l = res.max(), t['tier']
                        
                        if best_o < 0.75: best_l = "BLANK" # Below ore threshold
                        
                        clean_results[slot_id] = {'label': best_l, 'score': best_o, 'frame': buffer_files[idx]}
                        break
                if slot_id in clean_results: break

        # D. Visual Export
        # We find the 'best overall' clean frame to use for the right-side proof
        best_clean_idx = -1
        for res in clean_results.values():
            best_clean_idx = buffer_files.index(res['frame'])
            break
        
        if best_clean_idx != -1:
            proof_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[best_clean_idx]))
            for sid, data in clean_results.items():
                row, col = divmod(sid, 6)
                sx, sy = int(74+(col*59.1))-24, int(261+(row*59.1))-24
                cv2.rectangle(proof_bgr, (sx, sy), (sx+48, sy+48), (0, 255, 0), 2)
                cv2.putText(proof_bgr, f"RECOVERED: {data['label']}", (sx-10, sy+60), 0, 0.4, (0, 255, 0), 1)

            comparison = np.hstack((raw_bgr, proof_bgr))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"F{f_num}_Validation.jpg"), comparison)
            print(f" [+] Exported Forensic Recovery for Floor {f_num}")

if __name__ == "__main__":
    run_v59_visual_forensics()