import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import mss
import os
import time  # <--- The missing import
import pyautogui

# --- ADJUST THESE TWO VARIABLES ---
# Match these to the physical size of your game digits
CROP_W = 7  
CROP_H = 12 

SAVE_FOLDER = cfg.DIGIT_DIR
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

print("--- Digit Harvester v1.1 ---")
print(f"1. Hover mouse over a specific digit.")
print(f"2. Press 'S' to save.")
print(f"3. Press 'Q' to quit.")

with mss.mss() as sct:
    while True:
        # Get mouse position
        mx, my = pyautogui.position()

        # Capture a 40x40 area around mouse to create the preview
        grab_area = {
            'top': int(my - 20), 
            'left': int(mx - 20), 
            'width': 40, 
            'height': 40
        }
        
        # Grab screen and convert to Grayscale
        raw = np.array(sct.grab(grab_area))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        
        # Optional: Invert if your game has white text and you want 
        # to stick to the 'Black on White' standard
        # gray = cv2.bitwise_not(gray)

        # Calculate bounding box for the red viewfinder
        x1, y1 = 20 - (CROP_W//2), 20 - (CROP_H//2)
        x2, y2 = x1 + CROP_W, y1 + CROP_H

        # Create color preview window
        preview_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview_color, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # Zoom for precision
        zoomed = cv2.resize(preview_color, (250, 250), interpolation=cv2.INTER_NEAREST)
        cv2.putText(zoomed, f"Size: {CROP_W}x{CROP_H}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Capture Preview", zoomed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the raw grayscale crop
            digit_crop = gray[y1:y2, x1:x2]
            filename = f"{SAVE_FOLDER}/digit_{int(time.time()*100)}.png"
            cv2.imwrite(filename, digit_crop)
            print(f"Saved: {filename}")
            
        elif key == ord('q'):
            break

cv2.destroyAllWindows()