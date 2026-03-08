import cv2
import numpy as np
import os
import json
import sys
import shutil

# --- 1. CONFIGURATION ---
DATASETS = ["0", "1", "2", "3", "4"] 
DIGITS_DIR = "digits"
TEMPLATE_DIR = "templates"
BASE_EVIDENCE_DIR = "Pass1_Evidence"

# ROIs
HEADER_ROI = (56, 100, 16, 35)    
DIG_STAGE_ROI = (330, 185, 20, 100) 

def run_inspector_sentinel():
    print(f"--- INITIATING v38.3 INSPECTOR SENTINEL ---")
    
    # 1. Setup Assets
    digit_map = load_digit_map()
    if not os.path.exists(BASE_EVIDENCE_DIR): os.makedirs(BASE_EVIDENCE_DIR)

    # 2. Iterate through Datasets
    for ds_id in DATASETS:
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.isdir(buffer_path): continue
        
        evidence_path = os.path.join(BASE_EVIDENCE_DIR, f"Run_{ds_id}")
        if os.path.exists(evidence_path): shutil.rmtree(evidence_path)
        os.makedirs(evidence_path)
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        milestones = []
        current_f = 0
        
        print(f"\n[START] Processing Dataset {ds_id} ({len(frames)} frames)")
        
        for i, f_name in enumerate(frames):
            img = cv2.imread(os.path.join(buffer_path, f_name))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # SENSOR A & B
            h_val = get_bitwise_number(gray[56:72, 100:135], digit_map)
            d_val = get_bitwise_number(gray[330:350, 185:285], digit_map)
            read_val = max(h_val, d_val)
            
            # HALLUCINATION FILTER: No jumps > 10 floors allowed without high stability
            if read_val > current_f:
                if (read_val - current_f) > 10:
                    # Potential Hallucination (like 190). Ignore for now.
                    read_val = -1 
            
            if read_val > current_f:
                current_f = read_val
                entry = {'idx': i, 'floor': current_f, 'frame': f_name}
                milestones.append(entry)
                
                # REAL-TIME EVIDENCE EXPORT
                # Save the image immediately so user can validate
                cv2.imwrite(f"{evidence_path}/F{current_f}_{f_name}", img)
                
                print(f"\n [RUN {ds_id}] Milestone: Floor {current_f} at {f_name}")
            
            sys.stdout.write(f"\r Progress: {f_name} | Max Floor: {current_f} | Read: {read_val}")
            sys.stdout.flush()

        # Save individual run manifest
        with open(f"milestones_run_{ds_id}.json", 'w') as f:
            json.dump(milestones, f, indent=4)
            
        # PERFORM GAP AUDIT
        perform_gap_audit(ds_id, milestones, current_f)

    print(f"\n--- ALL RUNS COMPLETE ---")

def load_digit_map():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            v = int(f[0]); img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
            _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
            d_map[v].append(b)
    return d_map

def get_bitwise_number(gray_roi, digit_map, min_conf=0.88):
    _, bin_h = cv2.threshold(gray_roi, 195, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t in temps:
            res = cv2.matchTemplate(bin_h, t, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]): matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = [m['val'] for i, m in enumerate(matches) if i == 0 or abs(m['x'] - matches[i-1]['x']) > 6]
    return int("".join(map(str, unique))) if unique else -1

def perform_gap_audit(run_id, milestones, max_floor):
    found_floors = set(m['floor'] for m in milestones)
    all_floors = set(range(1, max_floor + 1))
    missing = sorted(list(all_floors - found_floors))
    print(f"\n--- GAP AUDIT FOR RUN {run_id} ---")
    print(f" Total Milestones: {len(milestones)}")
    print(f" Missing Floors: {missing if missing else 'None'}")

if __name__ == "__main__":
    run_inspector_sentinel()