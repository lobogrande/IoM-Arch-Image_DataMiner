import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2

# --- LOAD YOUR REFERENCE IMAGE ---
# Use the frame where "Dig Stage: 14" is most visible
image_path = "capture_buffer/frame_20260306_233455_764160.png" 
img = cv2.imread(image_path)

if img is None:
    print(f"!!! Error: Could not find {image_path}")
else:
    print("--- DIG SITE CALIBRATOR ---")
    print("1. Click and DRAG a box around the 'Dig Stage: XX' text.")
    print("2. Press 'ENTER' or 'SPACE' to confirm.")
    print("3. The terminal will print your NEW_DIG_SITE_ROI.")

    # This opens a native window that lets you select the region
    roi = cv2.selectROI("SELECT DIG SITE TEXT", img, fromCenter=False, showCrosshair=True)
    
    # OpenCV's selectROI returns (x, y, w, h)
    # Our script uses (y, x, h, w) for consistency with slicing
    x, y, w, h = roi
    print("\n" + "="*30)
    print(f"NEW_DIG_SITE_ROI = ({y}, {x}, {h}, {w})")
    print("="*30)
    
    cv2.destroyAllWindows()