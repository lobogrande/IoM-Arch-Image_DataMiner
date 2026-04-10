import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_PATH = cfg.get_buffer_path(TARGET_RUN)
OUTPUT_DIR = cfg.TEMPLATE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selection State
selection = None
dragging = False
roi_box = [340, 40, 40, 60] # Default starting [y, x, w, h]

def mouse_callback(event, x, y, flags, param):
    global selection, dragging, roi_box
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        selection = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        x1, y1 = selection
        roi_box = [min(y, y1), min(x, x1), abs(x - x1), abs(y - y1)]

def run_sprite_harvester(frame_idx):
    global roi_box
    buffer_files = sorted([f for f in os.listdir(BUFFER_PATH) if f.endswith(('.png', '.jpg'))])
    img = cv2.imread(os.path.join(BUFFER_PATH, buffer_files[frame_idx]))
    if img is None: return

    cv2.namedWindow("Sprite Harvester")
    cv2.setMouseCallback("Sprite Harvester", mouse_callback)

    print("\n--- SPRITE HARVESTER (v1.0) ---")
    print("MOUSE: Draw Box | WASD: Move Box | IJKL: Resize Box | ENTER: Save | ESC: Exit")

    while True:
        display_img = img.copy()
        y, x, w, h = roi_box
        
        # Draw the target box
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Crosshair for center verification
        cv2.line(display_img, (x + w//2, y), (x + w//2, y + h), (0, 255, 0), 1)
        cv2.line(display_img, (x, y + h//2), (x + w, y + h//2), (0, 255, 0), 1)

        cv2.imshow("Sprite Harvester", display_img)
        key = cv2.waitKey(1) & 0xFF

        # Movement
        if key == ord('w'): roi_box[0] -= 1
        elif key == ord('s'): roi_box[0] += 1
        elif key == ord('a'): roi_box[1] -= 1
        elif key == ord('d'): roi_box[1] += 1
        # Resizing
        elif key == ord('i'): roi_box[3] -= 1
        elif key == ord('k'): roi_box[3] += 1
        elif key == ord('j'): roi_box[2] -= 1
        elif key == ord('l'): roi_box[2] += 1
        
        elif key == 13: # ENTER: Save
            crop = img[y:y+h, x:x+w]
            # Prompt for name
            name = input("Enter filename (e.g., player_right.png): ")
            cv2.imwrite(os.path.join(OUTPUT_DIR, name), crop)
            print(f" [!] Saved {name} ({w}x{h})")
            
        elif key == 27: # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run on frame 0 or any frame where the player is stationary at 'Home'
    run_sprite_harvester(429) #235 - 429