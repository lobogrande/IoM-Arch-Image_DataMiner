import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np

def calibrate_relative_offset(image_path):
    img_gray = cv2.imread(image_path, 0)
    
    # 1. Slot 0 Center is known to be X=74, Y=350 (from our grid calibration)
    slot_0_x = 74
    
    # 2. Find Character Center in the left gutter
    search_area = img_gray[300:400, 0:100] # Narrower Y-band
    _, character_mask = cv2.threshold(search_area, 75, 255, cv2.THRESH_BINARY_INV)
    _, noise_mask = cv2.threshold(search_area, 20, 255, cv2.THRESH_BINARY_INV)
    final_mask = cv2.bitwise_and(character_mask, cv2.bitwise_not(noise_mask))
    
    coords = cv2.findNonZero(final_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        char_center_x = x + w//2
        offset = slot_0_x - char_center_x
        print(f"--- RELATIVE OFFSET CALIBRATED ---")
        print(f"Character Center X: {char_center_x}")
        print(f"Slot 0 Center X: {slot_0_x}")
        print(f"Required Left-Offset: {offset} pixels")
        return offset
    return None

# Run this on a frame where player is to the left of slot 0
calibrate_relative_offset(os.path.join(cfg.get_buffer_path(0), "frame_20260306_231745_717968.png"))