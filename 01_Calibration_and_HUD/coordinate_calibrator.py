import cv2
import os

# --- 1. PROBED COORDINATES ---
# Header ROI: Click 1 (105, 58) to Click 2 (127, 70)
HEADER_ROI = (58, 105, 12, 22)      # (Y, X, H, W)

# Dig Stage ROI: Click 3 (163, 233) to Click 4 (279, 246)
DIG_STAGE_ROI = (233, 163, 13, 116) # (Y, X, H, W)

# HUD Overlay: Based on your Probe for the purple box
HUD_X1, HUD_Y1 = 161, 231           # Top Left (+1px buffer)
HUD_X2, HUD_Y2 = 281, 248           # Bottom Right (+2px buffer)

# --- 2. INPUT IMAGE ---
TEST_IMAGE = "capture_buffer_0/frame_20260306_233844_294839.png" 
OUTPUT_DIR = "calibration_outputs_v2_1"

def run_calibration_v2_1():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"!!! Error: Could not find {TEST_IMAGE}")
        return

    # A. THE AI'S EYES (What the Worker actually scans)
    y_h, x_h, h_h, w_h = HEADER_ROI
    cv2.imwrite(f"{OUTPUT_DIR}/1_AI_Header_Crop.jpg", img[y_h:y_h+h_h, x_h:x_h+w_h])

    y_d, x_d, h_d, w_d = DIG_STAGE_ROI
    cv2.imwrite(f"{OUTPUT_DIR}/2_AI_DigStage_Crop.jpg", img[y_d:y_d+h_d, x_d:x_d+w_d])

    # B. THE HUD OVERLAY (The visual forensic output)
    hud_preview = img.copy()
    cv2.rectangle(hud_preview, (HUD_X1, HUD_Y1), (HUD_X2, HUD_Y2), (255, 0, 255), 1)
    
    # Context Image (Full Screen)
    cv2.imwrite(f"{OUTPUT_DIR}/3_HUD_Full_Context.jpg", hud_preview)
    
    # Zoomed Image (The precision check)
    # We crop a bit around the box to see the alignment clearly
    zoom_crop = hud_preview[HUD_Y1-30:HUD_Y2+30, HUD_X1-30:HUD_X2+30]
    cv2.imwrite(f"{OUTPUT_DIR}/4_HUD_Zoom_Check.jpg", zoom_crop)

    print(f"--- CALIBRATION V2.1 COMPLETE ---")
    print(f"Check '{OUTPUT_DIR}' for the 4-piece evidence set.")

if __name__ == "__main__":
    run_calibration_v2_1()