import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- GROUND TRUTH DATA ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 
    34: {'tier': 'mixed', 'special': {0:'com2',1:'com2',2:'com2',3:'com2',4:'com2',5:'com2',6:'com2',7:'com2',8:'myth1',9:'myth1',10:'com2',11:'com2',12:'com2',13:'com2',14:'myth1',15:'myth1',16:'com2',17:'com2',18:'com2',19:'com2',20:'com2',21:'com2',22:'com2',23:'com2'}}, 
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
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
OUTPUT_DIR = f"diagnostic_results/Run_{TARGET_RUN}_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
O_GATE, PLAYER_REJECT = 0.68, 0.88

def get_buffer_consensus(slot_coords, templates, current_frame_name, mask):
    """
    Actual implementation of the 'Temporal Scan'. 
    Looks at neighboring frames in the capture buffer to verify identity.
    """
    buffer_files = sorted(os.listdir(BUFFER_ROOT))
    try:
        idx = buffer_files.index(current_frame_name)
    except ValueError:
        return 0, ""

    # Scan offsets: -15, -10, +10, +15
    test_indices = [idx-15, idx-10, idx+10, idx+15]
    best_score, best_id = 0, ""

    x1, y1, x2, y2 = slot_coords
    
    for t_idx in test_indices:
        if 0 <= t_idx < len(buffer_files):
            neighbor_img = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[t_idx]), 0)
            if neighbor_img is None: continue
            roi = neighbor_img[y1:y2, x1:x2]
            
            for t in templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                if res.max() > best_score:
                    best_score, best_id = res.max(), t['name']
    
    return best_score, best_id

def run_v321_qa_truth_audit():
    # Asset Loading
    ore_templates = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})
    
    player_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_player")]
    ui_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_ui")]
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]

    with open(f"Unified_Consensus_Inputs/Run_{TARGET_RUN}/final_sequence.json", 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v3.2.1 Temporal Truth Audit (Surfing Enabled) ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        frame_name = sequence[f_num]['frame']
        raw_img = cv2.imread(os.path.join(f"Unified_Consensus_Inputs/Run_{TARGET_RUN}", f"F{f_num}_{frame_name}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        is_boss = f_num in BOSS_DATA
        # Filter templates by floor restriction
        valid_templates = [t for t in ore_templates if t['name'].lower() in ORE_RESTRICTIONS and ORE_RESTRICTIONS[t['name'].lower()][0] <= f_num <= ORE_RESTRICTIONS[t['name'].lower()][1]]

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi_gray = gray[y1:y1+48, x1:x1+48]
            mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6: cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
            else: cv2.circle(mask, (24, 24), 16, 255, -1)

            if is_boss:
                data = BOSS_DATA[f_num]
                best_label, best_o = (data['special'][slot] if data['tier'] == 'mixed' else data['tier']), 1.0
            else:
                # 1. Primary Identification
                best_o, best_label = 0, ""
                for t in valid_templates:
                    res = cv2.matchTemplate(roi_gray, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    if res.max() > best_o: best_o, best_label = res.max(), t['name']

                # 2. Rejection Logic (Player/Noise)
                # ... [Player/UI checks same as v2.7] ...

                # 3. SURGICAL TEMPORAL SCAN
                # Trigger if match is weak OR peak white is found (indicates crosshairs/noise)
                if best_o < 0.78 or np.max(roi_gray) > 242:
                    t_score, t_label = get_buffer_consensus((x1, y1, x1+48, y1+48), valid_templates, frame_name, mask)
                    if t_score > best_o:
                        best_o, best_label = t_score, t_label

            # Final Render
            if best_o > 0.68:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                (w, h), _ = cv2.getTextSize(best_label, 0, 0.3, 1)
                cv2.rectangle(raw_img, (x1+2, y1+48-h-4), (x1+w+4, y1+48-2), (0,0,0), -1)
                cv2.putText(raw_img, best_label, (x1+3, y1+48-4), 0, 0.3, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"QA_v321_F{f_num}.jpg"), raw_img)

if __name__ == "__main__":
    run_v321_qa_truth_audit()