import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CORRECTED GROUND TRUTH ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 
    34: {'tier': 'mixed', 'special': {0:'com3',1:'com3',2:'com3',3:'com3',4:'com3',5:'com3',6:'com3',7:'com3',8:'myth1',9:'myth1',10:'com3',11:'com3',12:'com3',13:'com3',14:'myth1',15:'myth1',16:'com3',17:'com3',18:'com3',19:'com3',20:'com3',21:'com3',22:'com3',23:'com3'}}, 
    35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 
    49: {'tier': 'mixed', 'special': {0:"dirt3",1:"dirt3",2:"dirt3",3:"dirt3",4:"dirt3",5:"dirt3",6:"com3",7:"com3",8:"com3",9:"com3",10:"com3",11:"com3",12:"rare3",13:"rare3",14:"rare3",15:"rare3",16:"rare3",17:"rare3",18:"myth2",19:"myth2",20:"myth2",21:"myth2",22:"myth2",23:"myth2"}},
    74: {'tier': 'mixed', 'special': {0:'dirt3',1:'dirt3',2:'dirt3',3:'dirt3',4:'dirt3',5:'dirt3',6:'dirt3',7:'dirt3',8:'dirt3',9:'dirt3',10:'dirt3',11:'dirt3',12:'dirt3',13:'dirt3',14:'dirt3',15:'dirt3',16:'dirt3',17:'dirt3',18:'dirt3',19:'dirt3',20:'div1',21:'div1',22:'dirt3',23:'dirt3'}}, 
    98: {'tier': 'myth3'}, 
    99: {'tier': 'mixed', 'special': {0:"com3",1:"rare3",2:"epic3",3:"leg3",4:"myth3",5:"div2",6:"com3",7:"rare3",8:"epic3",9:"leg3",10:"myth3",11:"div2",12:"com3",13:"rare3",14:"epic3",15:"leg3",16:"myth3",17:"div2",18:"com3",19:"rare3",20:"epic3",21:"leg3",22:"myth3",23:"div2"}}
}

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 51)
BUFFER_ROOT = "capture_buffer_0"
OUTPUT_DIR = f"diagnostic_results/Run_{TARGET_RUN}_v34_{datetime.now().strftime('%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE, O_GATE, P_GATE, U_GATE = 6, 0.68, 0.88, 0.82

def get_temporal_identity(coords, templates, frame_name, mask):
    files = sorted(os.listdir(BUFFER_ROOT))
    if frame_name not in files: return 0, ""
    idx = files.index(frame_name)
    best_s, best_n = 0, ""
    # Searching wider window for crosshairs/banners
    for off in [-15, -12, -10, 8, 10, 15]: 
        if 0 <= idx+off < len(files):
            img = cv2.imread(os.path.join(BUFFER_ROOT, files[idx+off]), 0)
            if img is None: continue
            roi = img[coords[1]:coords[3], coords[0]:coords[2]]
            for t in templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                if res.max() > best_s: best_s, best_n = res.max(), t['name']
    return best_s, best_n

def run_v34_restoration():
    # Asset Loading
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    player_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_player")]
    ui_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_ui")]
    all_ore_t = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None: all_ore_t.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(f"Unified_Consensus_Inputs/Run_{TARGET_RUN}/final_sequence.json", 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}

    suspicion_report = []

    for f_num in TARGET_FLOORS:
        if f_num not in seq: continue
        f_name = seq[f_num]['frame']
        raw_img = cv2.imread(os.path.join(f"Unified_Consensus_Inputs/Run_{TARGET_RUN}", f"F{f_num}_{f_name}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        is_boss = f_num in BOSS_DATA
        # CRITICAL: Restriction Filter
        valid_ore_t = [t for t in all_ore_t if ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[0] <= f_num <= ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[1]]

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray[y1:y1+48, x1:x1+48]
            ore_mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6: cv2.rectangle(ore_mask, (5, 18), (43, 45), 255, -1)
            else: cv2.circle(ore_mask, (24, 24), 16, 255, -1)

            # --- 1. OCCUPANCY ---
            if not is_boss and min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_t]) <= D_GATE: continue

            # --- 2. PLAYER (PURPLE) ---
            best_p = max([cv2.matchTemplate(roi, pt, cv2.TM_CCORR_NORMED).max() for pt in player_t] + [0])
            if not is_boss and best_p > P_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 0, 255), 1)
                continue

            # --- 3. IDENTIFICATION ---
            best_o, best_label = 0, ""
            if is_boss:
                data = BOSS_DATA[f_num]
                best_label = (data['special'][slot] if data['tier'] == 'mixed' else data['tier'])
                best_o = 1.0 # Forced Identity
            else:
                for t in valid_ore_t:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=ore_mask)
                    if res.max() > best_o: best_o, best_label = res.max(), t['name']

            # --- 4. UI REJECTION (CYAN) ---
            if not is_boss and slot < 6:
                best_u = max([cv2.matchTemplate(roi, ut, cv2.TM_CCORR_NORMED).max() for ui_t] + [0])
                # Only Cyan if it fails to find a high-conf ore
                if best_o < 0.82 and (best_u > U_GATE or np.max(roi[5:15, :]) > 242):
                    cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 255, 0), 1)
                    continue

            # --- 5. TEMPORAL SCAN ---
            # Triggered by crosshairs (high local intensity) or very weak matches
            if not is_boss and (best_o < 0.76 or (np.max(roi) > 235 and best_o < 0.90)):
                t_score, t_label = get_temporal_identity((x1, y1, x1+48, y1+48), valid_ore_t, f_name, ore_mask)
                if t_score > best_o: best_o, best_label = t_score, t_label

            # --- FINAL DRAW ---
            if best_o > O_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                (w, h), _ = cv2.getTextSize(best_label, 0, 0.3, 1)
                cv2.rectangle(raw_img, (x1+2, y1+48-h-4), (x1+w+4, y1+48-2), (0,0,0), -1)
                cv2.putText(raw_img, best_label, (x1+3, y1+48-4), 0, 0.3, (0, 255, 0), 1)
                
                if best_o < 0.75:
                    suspicion_report.append({"floor": f_num, "slot": slot, "label": best_label, "score": float(best_o)})

    with open(os.path.join(OUTPUT_DIR, "suspicion.json"), "w") as f:
        json.dump(suspicion_report, f, indent=4)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Restored_F{f_num}.jpg"), raw_img)

if __name__ == "__main__":
    run_v34_restoration()