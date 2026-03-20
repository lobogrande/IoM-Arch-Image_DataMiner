import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 7
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def calculate_texture_energy(roi):
    # The Laplacian measures the 'amount of edges' in an image
    # Ores usually have higher variance (energy) than noisy floor gravel
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    return laplacian.var()

def run_energy_profiler():
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    files = [f for f in os.listdir(run_path) if f.startswith(f"F{TARGET_FLOOR}_")]
    img = cv2.imread(os.path.join(run_path, files[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"--- TEXTURE ENERGY PROFILE: FLOOR {TARGET_FLOOR} ---")
    print("Goal: Find the 'Energy Gap' between Empty and Occupied slots.\n")

    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        roi = gray[cy-24:cy+24, cx-24:cx+24]
        
        energy = calculate_texture_energy(roi)
        
        # Determine Color for HUD
        # We will use 100 as a temporary 'Eye-Ball' threshold
        color = (0, 255, 0) if energy > 100 else (255, 0, 0)
        
        cv2.rectangle(img, (cx-24, cy-24), (cx+24, cy+24), color, 1)
        cv2.putText(img, f"E:{energy:.1f}", (cx-20, cy+20), 0, 0.35, color, 1)
        
        print(f"Slot {slot:02d} | Energy: {energy:.2f}")

    cv2.imwrite(f"Energy_Profile_F{TARGET_FLOOR}.jpg", img)
    print(f"\nSaved Energy_Profile_F{TARGET_FLOOR}.jpg. Check if the 'Ghost' slots have lower energy.")

if __name__ == "__main__":
    run_energy_profiler()