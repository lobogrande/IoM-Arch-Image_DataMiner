import cv2
import os

# --- 1. CURRENT "SENTINEL" COORDINATES ---
# Update these values to test new offsets
HEADER_ROI = (56, 100, 16, 35)      # (Y, X, H, W)
DIG_STAGE_ROI = (330, 185, 20, 100) # (Y, X, H, W) - The AI scanner

# HUD Overlay Coordinates (Top of grid area)
HUD_X1, HUD_Y1 = 160, 210           # Top Left
HUD_X2, HUD_Y2 = 300, 245           # Bottom Right

# --- 2. INPUT IMAGE ---
# Change this to any image you want to test from your buffer
TEST_IMAGE = "capture_buffer_0/frame_20260306_231742_176023.png" 
OUTPUT_DIR = "calibration_outputs"

def run_calibration_check():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"!!! Error: Could not find {TEST_IMAGE}")
        return

    # A. THE AI VIEW (What the bitwise engine sees)
    # Header ROI
    y, x, h, w = HEADER_ROI
    header_crop = img[y:y+h, x:x+w]
    cv2.imwrite(f"{OUTPUT_DIR}/view_AI_header.jpg", header_crop)

    # Dig Stage ROI
    y2, x2, h2, w2 = DIG_STAGE_ROI
    dig_stage_crop = img[y2:y2+h2, x2:x2+w2]
    cv2.imwrite(f"{OUTPUT_DIR}/view_AI_dig_stage.jpg", dig_stage_crop)

    # B. THE HUD VIEW (Where the purple box lands)
    hud_preview = img.copy()
    cv2.rectangle(hud_preview, (HUD_X1, HUD_Y1), (HUD_X2, HUD_Y2), (255, 0, 255), 2)
    
    # Save full image with HUD for context
    cv2.imwrite(f"{OUTPUT_DIR}/view_HUD_context.jpg", hud_preview)
    
    # Save a zoomed-in crop of just the HUD area
    hud_zoom = hud_preview[HUD_Y1-20:HUD_Y2+20, HUD_X1-20:HUD_X2+20]
    cv2.imwrite(f"{OUTPUT_DIR}/view_HUD_zoom.jpg", hud_zoom)

    print(f"--- CALIBRATION COMPLETE ---")
    print(f"Check the '{OUTPUT_DIR}' folder for:")
    print(f" 1. view_AI_header.jpg    <- Should contain ONLY the Stage number.")
    print(f" 2. view_AI_dig_stage.jpg <- Should contain ONLY the 'Dig Stage: XX' text.")
    print(f" 3. view_HUD_zoom.jpg     <- Shows if the purple box is framing the text correctly.")

if __name__ == "__main__":
    run_calibration_check()