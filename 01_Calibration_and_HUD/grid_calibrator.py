import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np

# --- 1. SET THE TEST IMAGE ---
IMAGE_SOURCE = "capture_buffer_0/frame_20260306_232957_356908.png" 

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        params['points'].append((x, y))
        print(f"Captured Point {len(params['points'])}: X={x}, Y={y}")
        cv2.circle(params['img'], (x, y), 4, (0, 255, 0), -1)
        cv2.imshow('GRID CALIBRATOR', params['img'])
        
        if len(params['points']) == 4:
            generate_grid_report(params['points'], params['raw_img'])

def generate_grid_report(pts, raw):
    # pts order: TL, TR, BL, BR
    tl, tr, bl, br = pts
    
    # Calculate Steps
    x_step = ((tr[0] - tl[0]) / 5 + (br[0] - bl[0]) / 5) / 2
    y_step = ((bl[1] - tl[1]) / 3 + (br[1] - tr[1]) / 3) / 2
    
    print(f"\n--- GRID CALCULATION COMPLETE ---")
    print(f"Start Point (Slot 1 Center): X={tl[0]}, Y={tl[1]}")
    print(f"Calculated X_STEP: {x_step:.2f}")
    print(f"Calculated Y_STEP: {y_step:.2f}")
    
    # Generate Verification Image
    verif = raw.copy()
    for row in range(4):
        for col in range(6):
            cx = int(tl[0] + (col * x_step))
            cy = int(tl[1] + (row * y_step))
            
            # AI Scanner Dot (Green)
            cv2.circle(verif, (cx, cy), 2, (0, 255, 0), -1)
            # HUD Box (Purple - 48x48)
            cv2.rectangle(verif, (cx-24, cy-24), (cx+24, cy+24), (255, 0, 255), 1)
            
    cv2.imwrite("grid_verification_check.jpg", verif)
    print("Verification image saved: 'grid_verification_check.jpg'")
    print("If boxes are aligned, press 'q' to exit and copy these numbers.")

def run_grid_probe():
    img = cv2.imread(IMAGE_SOURCE)
    if img is None: return

    print("\n--- INITIATING GRID CALIBRATION ---")
    print("Click the EXACT CENTER of these 4 ores in order:")
    print("1. Slot 1 (Top-Left corner)")
    print("2. Slot 6 (Top-Right corner)")
    print("3. Slot 19 (Bottom-Left corner)")
    print("4. Slot 24 (Bottom-Right corner)")

    params = {'img': img.copy(), 'raw_img': img, 'points': []}
    cv2.imshow('GRID CALIBRATOR', img)
    cv2.setMouseCallback('GRID CALIBRATOR', click_event, params)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_grid_probe()