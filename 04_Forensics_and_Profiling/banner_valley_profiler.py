import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED FROM YOUR PLOT ---
VALLEY_THRESHOLD = 15.0  # Blue line must stay below this to be "Banner-ish"
GAP_MAX = 40             # Max pixels of "noise/text" we will bridge over
MIN_BANNER_H = 35        # Total height of the bridged object

def detect_bridged_zones(intensities):
    # Create a mask of rows that are "dark enough"
    mask = (intensities < VALLEY_THRESHOLD).astype(np.uint8)
    
    # MORPHOLOGICAL CLOSING: This bridges the "text spikes"
    # A kernel of size 40 means any gap smaller than 40px will be filled
    kernel = np.ones((GAP_MAX, 1), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    zones = []
    start_y = None
    for y, val in enumerate(closed_mask):
        if val == 1 and start_y is None:
            start_y = y
        elif val == 0 and start_y is not None:
            height = y - start_y
            if height >= MIN_BANNER_H:
                zones.append((start_y, y))
            start_y = None
    return zones

# Next, we integrate this into the Sentinel Alpha loop to verify the red boxes...