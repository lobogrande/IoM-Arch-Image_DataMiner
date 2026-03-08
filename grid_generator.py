import cv2
import numpy as np
import mss
from pynput import mouse
import time

# --- CONFIGURATION ---
ROI = {'top': 225, 'left': 10, 'width': 446, 'height': 677}
COLUMNS = 6
ROWS = 4

# Global storage for clicks
clicks = []

def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.left:
        clicks.append((int(x), int(y)))
        if len(clicks) == 1:
            print(f"Captured Top-Left Center: {clicks[0]}")
            print("Now, click the CENTER of the BOTTOM-RIGHT ore (Slot 23).")
        elif len(clicks) == 2:
            print(f"Captured Bottom-Right Center: {clicks[1]}")
            return False # Stop the listener

def calculate_grid(tl, br):
    # Calculate the total distance between the centers of the corner ores
    total_dist_x = br[0] - tl[0]
    total_dist_y = br[1] - tl[1]
    
    # Calculate the 'stride' (distance between each ore center)
    # Since there are 3 cols, there are 2 gaps (COLUMNS - 1)
    # Since there are 8 rows, there are 7 gaps (ROWS - 1)
    stride_x = total_dist_x / (COLUMNS - 1)
    stride_y = total_dist_y / (ROWS - 1)
    
    centers = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            center_x = int(tl[0] + (c * stride_x))
            center_y = int(tl[1] + (r * stride_y))
            centers.append((center_x, center_y))
            
    return centers

print("--- Interactive 2-Point Calibrator ---")
print("1. Click the CENTER of the TOP-LEFT ore (Slot 0).")

# Start listening for mouse clicks
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

# Perform the math
SLOT_COORDS = calculate_grid(clicks[0], clicks[1])

# --- VERIFICATION ---
with mss.mss() as sct:
    screenshot = np.array(sct.grab(ROI))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    print("\n--- GRID COORDINATES (PASTE INTO MASTER LOGGER) ---")
    print(f"SLOT_COORDS = {SLOT_COORDS}")
    print("--------------------------------------------------")

    for i, (x, y) in enumerate(SLOT_COORDS):
        # Convert absolute screen coords to ROI-local coords for drawing
        local_x = x - ROI['left']
        local_y = y - ROI['top']
        
        cv2.drawMarker(frame, (local_x, local_y), (0, 255, 0), 
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        cv2.putText(frame, str(i), (local_x + 5, local_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite("interactive_grid_check.png", frame)
    print("\nVerification saved as 'interactive_grid_check.png'.")
    print("Verify if the markers are perfectly centered on the ores.")