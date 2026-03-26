import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import mss
import time
import os
from datetime import datetime

# --- CONFIG ---
GAME_ROI = {'top': 225, 'left': 10, 'width': 446, 'height': 677}
HEADER_ROI = {'top': 281, 'left': 110, 'width': 35, 'height': 16}
BUFFER_DIR = "capture_buffer_4"

if not os.path.exists(BUFFER_DIR): os.makedirs(BUFFER_DIR)

print("\n" + "="*35 + "\n PRO LOGGER v12.0: HIGH-SPEED RECORDER \n" + "="*35)
print(" This script records frames as fast as possible for offline audit.")
print(" Press 'q' to stop recording and prepare for data extraction.")

frame_count = 0
start_time = time.time()

with mss.mss() as sct:
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Capture the full game area
            sct_img = sct.grab(GAME_ROI)
            frame = np.array(sct_img)
            
            # Save frame with a high-res timestamp for sequential processing
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"{BUFFER_DIR}/frame_{ts}.png"
            cv2.imwrite(fname, frame)
            
            frame_count += 1
            elapsed = time.time() - start_time
            
            # Simplified terminal output to keep Hz high
            if frame_count % 10 == 0:
                print(f"\rCaptured: {frame_count} frames | Time: {elapsed:.1f}s | Hz: {1.0/(time.perf_counter()-loop_start):.1f}", end="")
            
            # Short sleep to prevent CPU thermal throttling on long runs
            time.sleep(0.01) 

    except KeyboardInterrupt:
        pass

print(f"\n\nRecording Stopped. Total Frames: {frame_count}")
print(f"Stored in: {os.path.abspath(BUFFER_DIR)}")
print("Would you like the 'Auditor' script to scan these files for ores and floors now?")