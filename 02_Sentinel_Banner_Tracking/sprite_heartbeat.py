# sprite_heartbeat.py
# Run this to see what the 'Best' score actually is.
import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def check_heartbeat():
    raw_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_r = raw_r[int(raw_r.shape[0]*0.4):, :] # Behead
    h_r, w_r = tpl_r.shape
    
    # Target Slot 0 ROI
    tx, ty = int(66 - 55 - (w_r // 2)), int(249 - (h_r // 2))
    
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:500]
    max_seen = 0
    
    for f in files:
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), f), 0)
        roi = img[max(0, ty):ty+h_r+10, max(0, tx):tx+w_r+10]
        if roi.shape[0] < h_r: continue
        res = cv2.matchTemplate(roi, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > max_seen: max_seen = val
    
    print(f"DIAGNOSTIC: Highest confidence seen for Slot 0 body: {round(max_seen, 4)}")

if __name__ == "__main__":
    check_heartbeat()