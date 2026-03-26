import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np

# --- 1. SET THE TEST IMAGE ---
# Use the currated 3-digit floor image here
IMAGE_SOURCE = os.path.join(cfg.get_buffer_path(0), "frame_20260306_233844_294839.png") 

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Prints X,Y to terminal and draws a temporary red dot
        print(f"Clicked: X={x}, Y={y}")
        cv2.circle(params['img'], (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('PIXEL PROBE', params['img'])

def run_probe():
    img = cv2.imread(IMAGE_SOURCE)
    if img is None:
        print(f"Error: Could not find {IMAGE_SOURCE}")
        return

    print("\n--- INITIATING PIXEL PROBE ---")
    print("1. Click the TOP-LEFT of the [Stage: XX] number (Header).")
    print("2. Click the BOTTOM-RIGHT of the [Stage: XX] number (Header).")
    print("3. Click the TOP-LEFT of the [Dig Site: XX] text (Top of Grid).")
    print("4. Click the BOTTOM-RIGHT of the [Dig Site: XX] text (Top of Grid).")
    print("5. Click the CENTER of Slot 1 (Top-Left ore slot).")
    print("Press 'q' or ESC to exit when you have your numbers.")

    params = {'img': img}
    cv2.imshow('PIXEL PROBE', img)
    # Register the mouse callback function
    cv2.setMouseCallback('PIXEL PROBE', click_event, params)
    
    # Wait until 'q' or ESC is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_probe()