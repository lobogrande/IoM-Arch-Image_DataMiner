# sprite_sequencer.py
# Version: 2.8
# Purpose: Step 1 Master Sequencer for high-precision mining event detection.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.8"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# --- PRODUCTION CONSTANTS ---
ORE0_HUD_X, ORE0_HUD_Y = 72, 255
STEP = 59.0
PLAYER_OFFSET = 41.0
TPL_W, TPL_H = 40, 60

# --- STAIRCASE THRESHOLD MAP ---
STAIRCASE = {
    0: 0.90, 1: 0.85, 2: 0.82, 3: 0.78, 4: 0.75, 5: 0.72, 11: 0.82
}

def get_slot_geometry(slot_id):
    """Calculates all anchor points for a specific slot."""
    col = slot_id % 6
    row = 0 if slot_id < 6 else 1
    
    # 1. HUD Ore Center
    hox = int(ORE0_HUD_X + (col * STEP))
    hoy = int(ORE0_HUD_Y + (row * STEP))
    
    # 2. HUD Player Center (Visual Torso)
    # Slot 0-5 stands left of ore; Slot 11 stands right of ore
    hpx = int(hox - PLAYER_OFFSET) if slot_id < 6 else int(hox + PLAYER_OFFSET)
    hpy = hoy
    
    # 3. AI Top-Left (Template Anchor)
    # Offsets the center by half template width/height
    apx = int(hpx - (TPL_W // 2))
    apy = int(hpy - (TPL_H // 2))
    
    return (hox, hoy), (hpx, hpy), (apx, apy)

def run_production_scan():
    print(f"--- MASTER SEQUENCER v{VERSION} ---")
    
    # Load Templates
    full_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    full_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if full_r is None or full_l is None:
        print("Error: Missing player templates.")
        return
    
    bot_r, bot_l = full_r[30:, :], full_l[30:, :]
    bh, bw = bot_r.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    counts = Counter()

    # Pre-calculate slot targets
    targets = []
    for s_id in sorted(STAIRCASE.keys()):
        h_ore, h_play, a_play = get_slot_geometry(s_id)
        targets.append({
            'id': s_id,
            'ai_x': a_play[0], 'ai_y': a_play[1],
            'hud_px': h_play[0], 'hud_py': h_play[1],
            'hud_ox': h_ore[0], 'hud_oy': h_ore[1],
            'tpl_f': full_r if s_id < 6 else full_l,
            'tpl_b': bot_r if s_id < 6 else bot_l,
            'thresh': STAIRCASE[s_id]
        })

    print(f"Scanning {len(files)} frames using Hybrid Redundancy + Staircase Thresholds...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        for t in targets:
            # ROI Slicing (Ensure within image bounds)
            if t['ai_x'] < 0 or t['ai_y'] < 0 or t['ai_x']+40 > iw or t['ai_y']+60 > ih:
                continue
                
            # Full Body Match
            roi_f = img[t['ai_y'] : t['ai_y']+60, t['ai_x'] : t['ai_x']+40]
            res_f = cv2.matchTemplate(roi_f, t['tpl_f'], cv2.TM_CCOEFF_NORMED)
            score_f = cv2.minMaxLoc(res_f)[1]
            
            # Bottom Half Match (AI_Y + 30)
            roi_b = img[t['ai_y']+30 : t['ai_y']+60, t['ai_x'] : t['ai_x']+40]
            res_b = cv2.matchTemplate(roi_b, t['tpl_b'], cv2.TM_CCOEFF_NORMED)
            score_b = cv2.minMaxLoc(res_b)[1]

            best_val = max(score_f, score_b)

            if best_val >= t['thresh']:
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': t['id'],
                    'confidence': round(best_val, 4),
                    'method': 'Full' if score_f >= score_b else 'Bottom',
                    'ai_px': t['ai_x'],
                    'ai_py': t['ai_y'],
                    'hud_px': t['hud_px'],
                    'hud_py': t['hud_py'],
                    'hud_ox': t['hud_ox'],
                    'hud_oy': t['hud_oy']
                })
                counts[t['id']] += 1
                break 

        if f_idx % 2000 == 0:
            stat_line = " | ".join([f"S{s}:{counts[s]}" for s in sorted(counts.keys())])
            print(f"  [{f_idx:05d}] {stat_line if stat_line else 'Searching...'}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(results)} detections to {OUT_CSV}")
    print(f"Final Coverage: {dict(counts)}")

if __name__ == "__main__":
    run_production_scan()