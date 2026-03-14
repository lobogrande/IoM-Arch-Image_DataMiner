import cv2
import numpy as np

def calibrate_character_home(image_path):
    # Load frame in grayscale and color
    img_gray = cv2.imread(image_path, 0)
    img_bgr = cv2.imread(image_path)
    
    # 1. Define Search Area: The left gutter (X: 0-100, Y: 200-500)
    search_area = img_gray[200:500, 0:100]
    
    # 2. Character Signature: The sprite is dark/grey. 
    # We look for pixels in the 30-70 range.
    _, character_mask = cv2.threshold(search_area, 75, 255, cv2.THRESH_BINARY_INV)
    _, noise_mask = cv2.threshold(search_area, 20, 255, cv2.THRESH_BINARY_INV)
    final_mask = cv2.bitwise_and(character_mask, cv2.bitwise_not(noise_mask))
    
    # 3. Find the largest cluster (The Character)
    coords = cv2.findNonZero(final_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Adjust for the ROI offset
        abs_x, abs_y = x, y + 200
        print(f"--- CALIBRATION RESULTS ---")
        print(f"Character Home Box: [Y:{abs_y}:{abs_y+h}, X:{abs_x}:{abs_x+w}]")
        print(f"Center Point: ({abs_x + w//2}, {abs_y + h//2})")
        
        # Save a visual confirmation
        cv2.rectangle(img_bgr, (abs_x, abs_y), (abs_x+w, abs_y+h), (0, 255, 0), 2)
        cv2.imwrite("diagnostic_character_calibration.jpg", img_bgr)
    else:
        print("Character not found in search area.")

calibrate_character_home("capture_buffer_0/frame_00000.png")