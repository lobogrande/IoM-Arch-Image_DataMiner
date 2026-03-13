import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- GROUND TRUTH DATA ---
# Using the verified Boss and Restriction data to ensure we aren't chasing ghosts
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 
    34: {'tier': 'mixed', 'special': {0:'com3',1:'com3',2:'com3',3:'com3',4:'com3',5:'com3',6:'com3',7:'com3',8:'myth1',9:'myth1',10:'com3',11:'com3',12:'com3',13:'com3',14:'myth1',15:'myth1',16:'com3',17:'com3',18:'com3',19:'com3',20:'com3',21:'com3',22:'com3',23:'com3'}}, 
    35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}
}

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- TARGETED CONFIG ---
TARGET_RUN = "0"
PROBLEM_FLOORS = [2, 5, 6, 14, 24, 37] # Just Issue 1 floors
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
OUTPUT_DIR = f"diagnostic_results/Issue1_V361_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 6
O_GATE_DEFAULT = 0.68
O_GATE_TOP_ROW = 0.78  # Significantly higher bar for the HUD zone
DELTA_GATE_TOP_ROW = 0.15 # Ore must be much 'louder' than the background gravel

def run_issue1_isolated_audit():
    # 1. Asset Loading
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    ui_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_ui")]
    
    all_ore_t = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            all_ore_t.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running Issue 1 Audit v3.6.1 (Top-Row Isolation) ---")

    for f_num in PROBLEM_FLOORS:
        if f_num not in sequence: continue
        f_name = sequence[f_num]['frame']
        raw_img = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{f_name}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        is_boss = f_num in BOSS_DATA
        valid_templates = [t for t in all_ore_t if ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[0] <= f_num <= ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[1]]

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray[y1:y1+48, x1:x1+48]
            
            # --- GATE 1: OCCUPANCY ---
            diff = min([np.sum(cv2.absdiff(roi, bg)) / (2304) for bg in bg_t])
            if not is_boss and diff <= D_GATE: continue

            # --- GATE 2: TOP-ROW HARD EXCLUSION ---
            if not is_boss and slot < 6:
                # 2a. Peak-White Intensity Count (HUD Text Signature)
                white_pixels = np.sum(roi[5:18, :] > 245)
                # 2b. Negative UI Template Match
                best_u = max([cv2.matchTemplate(roi, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_t] + [0])
                
                if white_pixels > 12 or best_u > 0.85:
                    cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 255, 0), 1) # Cyan Reject
                    continue

            # --- GATE 3: IDENTITY SEARCH ---
            mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6: cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
            else: cv2.circle(mask, (24, 24), 16, 255, -1)

            best_o, best_label = 0, ""
            for t in valid_templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                if res.max() > best_o: best_o, best_label = res.max(), t['name']

            # --- GATE 4: FINAL VERDICT ---
            bg_match = max([cv2.matchTemplate(roi, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_t])
            current_o_gate = O_GATE_TOP_ROW if slot < 6 else O_GATE_DEFAULT
            current_delta_gate = DELTA_GATE_TOP_ROW if slot < 6 else 0.05
            
            if best_o > current_o_gate and (best_o - bg_match > current_delta_gate):
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                cv2.putText(raw_img, f"{best_label} ({best_o:.2f})", (x1+3, y1+44), 0, 0.3, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Issue1_F{f_num}.jpg"), raw_img)

if __name__ == "__main__":
    run_issue1_isolated_audit()