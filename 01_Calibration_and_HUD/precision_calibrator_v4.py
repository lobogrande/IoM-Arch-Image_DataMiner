import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import os

# --- 1. SET YOUR NEW PROBED NUMBERS HERE ---
# Use the Pixel Probe to find these exact points
# HUD (Wide): Top-Left of 'D' to Bottom-Right of the last digit
HUD_X1, HUD_Y1 = 161, 231           
HUD_X2, HUD_Y2 = 281, 248           

# AI (Narrow): Top-Left of the FIRST DIGIT to Bottom-Right of the LAST DIGIT
# This is what you are recalibrating now
AI_DIG_X1, AI_DIG_Y1 = 255, 230     # Placeholder (Update these!)
AI_DIG_X2, AI_DIG_Y2 = 282, 246     # Placeholder (Update these!)

# --- 2. INPUT IMAGE ---
TEST_IMAGE = os.path.join(cfg.get_buffer_path(0), "frame_20260306_233844_294839.png") 
OUTPUT_DIR = "calibration_v4_outputs"

def run_precision_audit():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    img = cv2.imread(TEST_IMAGE)
    if img is None: return

    # A. THE AI VIEW (The "Eyes" - Narrow)
    # We crop specifically what the bitwise engine will see
    ai_crop = img[AI_DIG_Y1:AI_DIG_Y2, AI_DIG_X1:AI_DIG_X2]
    cv2.imwrite(f"{OUTPUT_DIR}/1_AI_Eyes_Recalibrated.jpg", ai_crop)

    # B. THE HUD VIEW (The "Overlay" - Wide)
    # We draw the box on a context image
    hud_preview = img.copy()
    # Wide purple box for human verification
    cv2.rectangle(hud_preview, (HUD_X1, HUD_Y1), (HUD_X2, HUD_Y2), (255, 0, 255), 1)
    # Temporary thin green box to see the AI ROI inside it
    cv2.rectangle(hud_preview, (AI_DIG_X1, AI_DIG_Y1), (AI_DIG_X2, AI_DIG_Y2), (0, 255, 0), 1)
    
    cv2.imwrite(f"{OUTPUT_DIR}/2_HUD_Context_Audit.jpg", hud_preview)
    
    # Zoomed verification
    zoom = hud_preview[HUD_Y1-20:HUD_Y2+20, HUD_X1-20:HUD_X2+20]
    cv2.imwrite(f"{OUTPUT_DIR}/3_Zoom_Alignment_Check.jpg", zoom)

    print(f"--- PRECISION AUDIT COMPLETE ---")
    print(f"Check 1_AI_Eyes_Recalibrated.jpg: It should contain ONLY numbers.")
    print(f"Check 3_Zoom_Alignment_Check.jpg: Purple = HUD, Green = AI Zone.")

if __name__ == "__main__":
    run_precision_audit()