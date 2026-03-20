# sprite_reconciliation_audit.py
# Purpose: Reconcile Point (HUD) vs Pixel (AI) coordinate systems.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DIAG_OUT = "coordinate_audit_results.jpg"

def run_reconciliation():
    # 1. Load Templates
    player_tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    # Behead for stability
    player_tpl = player_tpl[int(player_tpl.shape[0]*0.4):, :]

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    print("--- COORDINATE RECONCILIATION AUDIT ---")
    
    # We will look at a frame where we KNOW Slot 0 is happening (e.g. index 64)
    # But to be safe, we'll scan the first 100 frames
    for f_idx in range(60, 70):
        img_path = os.path.join(cfg.get_buffer_path(0), files[f_idx])
        img = cv2.imread(img_path, 0)
        if img is None: continue
        
        # GLOBAL SEARCH for Player
        res_p = cv2.matchTemplate(img, player_tpl, cv2.TM_CCOEFF_NORMED)
        _, p_val, _, p_loc = cv2.minMaxLoc(res_p)
        
        if p_val > 0.85:
            # We found the player! Let's see their actual pixel coordinate
            actual_x = p_loc[0]
            actual_y = p_loc[1]
            
            # Now let's see where our 'HUD' thought the player should be (74, 261)
            # based on current v1.8 logic
            hud_x = 74 - 55 # Slot start - offset
            hud_y = 261
            
            print(f"Frame {f_idx}:")
            print(f"  [AI Detection] Pixel X: {actual_x}, Pixel Y: {actual_y}")
            print(f"  [HUD Logic]    Point X: {hud_x}, Point Y: {hud_y}")
            print(f"  [DISCREPANCY]  X-Delta: {actual_x - hud_x}, Y-Delta: {actual_y - hud_y}")
            
            # Save a visual proof
            proof = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(proof, (actual_x, actual_y), (actual_x+player_tpl.shape[1], actual_y+player_tpl.shape[0]), (0, 255, 0), 2)
            cv2.putText(proof, "AI DETECTED PLAYER", (actual_x, actual_y-10), 0, 0.7, (0, 255, 0), 2)
            cv2.imwrite(DIAG_OUT, proof)
            break

if __name__ == "__main__":
    run_reconciliation()