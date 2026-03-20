import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2

# Load your standard start frame
img = cv2.imread('capture_buffer/frame_20260306_233844_240227.png')
clone = img.copy()

def get_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"PIXEL CLICKED: X={x}, Y={y}")
        # Draw a temporary dot to show where you clicked
        cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("CALIBRATOR", clone)

print("--- GRID CALIBRATOR ---")
print("1. Click the EXACT CENTER of the Top-Left Ore (Slot 0).")
print("2. Click the EXACT CENTER of the Bottom-Right Ore (Slot 23).")
cv2.imshow("CALIBRATOR", img)
cv2.setMouseCallback("CALIBRATOR", get_pixel)
cv2.waitKey(0)
cv2.destroyAllWindows()