import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import csv
import sys
import threading
import time
from queue import Queue

# --- 1. CONFIGURATION ---
BUFFER_DIR = "capture_buffer"
TEMPLATE_DIR = cfg.TEMPLATE_DIR
DIGITS_DIR = cfg.DIGIT_DIR 
RECOVERY_DIR = "forensic_v34_1"
CSV_FILE = "FINAL_TOTAL_AUDIT_v34_1.csv"

START_IMAGE = "frame_20260306_231742_176023.png"
THRESHOLD = 0.5 

# --- 2. GLOBAL STATE & LOCKS ---
STATE = {
    'floor': 0,
    'is_running': True
}
job_queue = Queue()
csv_lock = threading.Lock() # Prevents the Traceback crash

# --- 3. COORDINATES ---
HEADER_ROI = (56, 100, 16, 35)
AI_COORDS = {i: (100 + (i % 6) * 60, 500 + (i // 6) * 65) for i in range(24)}

# --- 4. THE SCOUT (Producer) ---
def scout_run(frames, start_idx, digit_map):
    history = []
    
    for i in range(start_idx, len(frames)):
        img = cv2.imread(os.path.join(BUFFER_DIR, frames[i]))
        if img is None: continue
        
        roi_h = cv2.cvtColor(img[56:72, 100:135], cv2.COLOR_BGR2GRAY)
        h_val = get_bitwise_floor(roi_h, digit_map)
        
        # Stability check: 2 frames of consensus
        history.append(h_val)
        if len(history) > 2: history.pop(0)
        stable_val = history[0] if len(set(history)) == 1 else -1

        if stable_val > STATE['floor'] and stable_val != -1:
            with csv_lock: # Safely print match
                sys.stdout.write('\033[2K\r') # Clear the telemetry line
                print(f"[SCOUT] MATCH: Floor {stable_val} at {frames[i]}")
            STATE['floor'] = stable_val
            job_queue.put({'frame': frames[i], 'floor': stable_val})

        # Telemetry with padding to prevent "1818" artifacts
        msg = f" Scout: {frames[i]} | Read: {h_val} | Brain: {STATE['floor']}"
        sys.stdout.write(f"\r{msg}{' ' * 20}")
        sys.stdout.flush()
    
    STATE['is_running'] = False

# --- 5. THE AUDITOR (Consumer) ---
def auditor_run(templates):
    if not os.path.exists(RECOVERY_DIR): os.makedirs(RECOVERY_DIR)
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Floor", "Slot", "Tier"])
        
        while STATE['is_running'] or not job_queue.empty():
            if job_queue.empty(): time.sleep(0.01); continue
            
            job = job_queue.get()
            img = cv2.imread(os.path.join(BUFFER_DIR, job['frame']))
            if img is None: job_queue.task_done(); continue
            
            ores, hud = process_ores(img, job['floor'], templates)
            
            with csv_lock: # Atomic write to CSV
                for s, d in ores.items():
                    writer.writerow([job['frame'], job['floor'], s, d])
            
            cv2.imwrite(f"{RECOVERY_DIR}/F{job['floor']}_Audit.jpg", hud)
            with csv_lock:
                sys.stdout.write('\033[2K\r') # Clear telemetry for log
                print(f" [AUDITOR] Logged Floor {job['floor']}")
            job_queue.task_done()

# --- 6. CORE UTILS ---
def get_bitwise_floor(gray_h, digit_map):
    # Strict binary thresholding to kill ghost digits
    _, bin_h = cv2.threshold(gray_h, 200, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t in temps:
            res = cv2.matchTemplate(bin_h, t, cv2.TM_CCOEFF_NORMED)
            if res.max() > 0.88: # Tightened threshold
                locs = np.where(res >= 0.88)
                for pt in zip(*locs[::-1]):
                    matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = []
    if matches:
        unique.append(matches[0]['val'])
        for i in range(1, len(matches)):
            if abs(matches[i]['x'] - matches[i-1]['x']) > 6:
                unique.append(matches[i]['val'])
    return int("".join(map(str, unique))) if unique else -1

def process_ores(img, floor, templates):
    # Standard identification engine
    res = {}; gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(24):
        ax, ay = AI_COORDS[i]
        roi = gray[ay-24:ay+24, ax-24:ax+24]
        best_t, best_s = "obscured", 0.0
        for name, t_img in templates.items():
            m = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED).max()
            if m > best_s: best_s, best_t = m, name
        res[i] = best_t if best_s >= THRESHOLD else "obscured"
        cv2.rectangle(img, (ax-24, ay-24), (ax+24, ay+24), (0,255,255), 1)
    return res, img

if __name__ == "__main__":
    # Asset loading... [Standard logic for digit_map and templates]
    digit_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            v = int(f[0]); d_img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
            _, b = cv2.threshold(d_img, 200, 255, cv2.THRESH_BINARY); digit_map[v].append(b)
    
    templates = {f.split('.')[0]: cv2.imread(os.path.join(TEMPLATE_DIR, f), 0) for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')}
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(('.png', '.jpg'))])
    idx = frames.index(START_IMAGE)

    t1 = threading.Thread(target=scout_run, args=(frames, idx, digit_map), daemon=True)
    t1.start()
    auditor_run(templates)
    print("\n--- AUDIT COMPLETE ---")