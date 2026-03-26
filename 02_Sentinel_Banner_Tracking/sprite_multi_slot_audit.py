# sprite_multi_slot_audit.py
# Version: 1.0
# Purpose: Track 7 independent "Virtual Sensors" to diagnose Slot 4/5 blindness.

import sys, os, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# EXPERIMENT CONSTANTS (Locked in from Audits)
S0_X, S0_Y = 11, 249
STEP_X, STEP_Y = 117.2, 59.0
TEST_FRAMES = 5000 # Let's look deeper into the dataset

def run_multi_slot_audit():
    print(f"--- MULTI-SLOT SIGNAL AUDIT (First {TEST_FRAMES} Frames) ---")
    
    # 1. Load Templates (We use the full template for maximum features)
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_r is None or tpl_l is None: return
    th, tw = tpl_r.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:TEST_FRAMES]
    
    # Setup Sensors for Slots 0, 1, 2, 3, 4, 5, and 11
    sensors = []
    for s in range(6):
        cx = int(S0_X + (s * STEP_X))
        sensors.append({'id': s, 'box': (cx - 10, S0_Y - 20, tw + 20, th + 40), 'tpl': tpl_r})
    
    # Sensor for Slot 11 (Facing Left)
    cx11 = int(S0_X + (5 * STEP_X))
    sensors.append({'id': 11, 'box': (cx11 - 10, int(S0_Y + STEP_Y) - 20, tw + 20, th + 40), 'tpl': tpl_l})

    all_signals = {s['id']: [] for s in sensors}

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue

        for s in sensors:
            x, y, w, h = s['box']
            roi = img[max(0, y):y+h, max(0, x):x+w]
            
            # Simple Normalized Matching (The "Ground Truth" Signal)
            if roi.shape[0] >= s['tpl'].shape[0] and roi.shape[1] >= s['tpl'].shape[1]:
                res = cv2.matchTemplate(roi, s['tpl'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                all_signals[s['id']].append(max_val)
            else:
                all_signals[s['id']].append(0)

        if f_idx % 1000 == 0:
            print(f"  Audit: {f_idx}/{TEST_FRAMES} frames...")

    # 2. Plot the 7-Channel Signal Map
    plt.figure(figsize=(15, 8))
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
    for i, s_id in enumerate(all_signals.keys()):
        plt.plot(all_signals[s_id], label=f"Slot {s_id}", color=colors[i], alpha=0.7)
    
    plt.axhline(y=0.80, color='grey', linestyle='--', label='Threshold Floor')
    plt.title("Sprite Detection Confidence by Slot (Ground Truth Audit)")
    plt.xlabel("Frame Index")
    plt.ylabel("Confidence Score")
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig("multi_slot_audit_signal.png")
    
    print("\n[DONE] Audit complete. Check 'multi_slot_audit_signal.png' for the signal map.")

if __name__ == "__main__":
    run_multi_slot_audit()