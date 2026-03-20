# sprite_sequencer.py
# Version: 2.2
# Refactor: Per-slot terminal reporting and full-strip discovery logic.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.2"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CALIBRATED ANCHORS
R1_Y, R2_Y = 249, 308 # Calibrated Pixel Y for Row 1 and Row 2
MATCH_THRESHOLD = 0.78 # Lowered slightly for discovery phase

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Diagnostic Discovery) ---")
    
    # 1. Load Templates (No Masking for Discovery - we want raw peaks)
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_r is None or tpl_l is None: return
    
    h_r, w_r = tpl_r.shape
    h_l, w_l = tpl_l.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    slot_counts = Counter()

    print(f"Scanning {len(files)} frames for all slots...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        # --- ROW 1 DISCOVERY (Slots 0-5) ---
        # We scan a 100px vertical strip across the entire screen
        r1_strip = img[max(0, R1_Y-50):min(ih, R1_Y+50), :]
        res_r = cv2.matchTemplate(r1_strip, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_r)

        if max_val_r >= MATCH_THRESHOLD:
            # Calculate actual slot from the detected pixel
            # Slot center = detected_x + half_width + wait_offset
            # (Wait offset 55 is our guess - let's see where they land)
            found_x = max_loc_r[0] + (w_r // 2) + 55
            inferred_slot = round((found_x - 66) / 118.0)
            
            if 0 <= inferred_slot <= 5:
                results.append({'frame': f_idx, 'slot': inferred_slot, 'x': max_loc_r[0], 'conf': round(max_val_r, 4)})
                slot_counts[inferred_slot] += 1

        # --- ROW 2 DISCOVERY (Slot 11) ---
        r2_strip = img[max(0, R2_Y-50):min(ih, R2_Y+50), :]
        res_l = cv2.matchTemplate(r2_strip, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_l)

        if max_val_l >= MATCH_THRESHOLD:
            found_x_l = (max_loc_l[0] + (w_l // 2)) - 55
            inferred_slot_l = round((found_x_l - 66) / 118.0)
            
            if inferred_slot_l == 11:
                results.append({'frame': f_idx, 'slot': 11, 'x': max_loc_l[0], 'conf': round(max_val_l, 4)})
                slot_counts[11] += 1

        if f_idx % 2000 == 0:
            stats = ", ".join([f"S{s}:{c}" for s, c in sorted(slot_counts.items())])
            print(f"  [Progress] {f_idx}/{len(files)} | Hits: {stats if stats else 'None'}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} Summary: {dict(slot_counts)}")

if __name__ == "__main__":
    run_sprite_sequencer()