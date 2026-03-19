import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_theta_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# SEARCH AREA (User calibrated to the 40-450 game zone)
SCAN_Y_START = 40
SCAN_Y_END = 450
BANNER_H = 45

# MOTION PARAMETERS
# The banner moves up (decreases Y) by roughly 8-12 pixels per frame
MIN_VELOCITY = 5   
MAX_VELOCITY = 18  
COAST_LIMIT = 8    # How many frames to "guess" the path if the signal is lost

class TrajectorySentinel:
    def __init__(self):
        self.lock = None # {'y_top': y, 'lost_count': 0, 'velocity': 10}
        self.stationary_blacklist = set()

    def find_top_candidates(self, intensities):
        """Finds the 5 darkest 45px windows in the scan zone."""
        candidates = []
        for y in range(SCAN_Y_START, SCAN_Y_END - BANNER_H):
            avg = np.mean(intensities[y : y + BANNER_H])
            candidates.append({'y_top': y, 'avg': avg})
        # Sort by intensity (darkest first)
        candidates.sort(key=lambda x: x['avg'])
        return candidates[:8]

    def update(self, img_gray):
        h, w = img_gray.shape
        # Focus on center strip to avoid side-ore noise
        c1, c2 = int(w * 0.35), int(w * 0.65)
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        
        candidates = self.find_top_candidates(intensities)
        
        if self.lock is None:
            # SEARCH MODE: Look for a dark window that appears in the nucleation zone (Y > 250)
            for cand in candidates:
                if cand['y_top'] > 250 and cand['avg'] < 42.0:
                    self.lock = {'y_top': cand['y_top'], 'lost_count': 0, 'velocity': 10}
                    return self.lock, intensities
            return None, intensities
        else:
            # TRACK MODE: Find the candidate that moved UP at the correct speed
            best_match = None
            min_err = 999
            
            for cand in candidates:
                velocity = self.lock['y_top'] - cand['y_top']
                if MIN_VELOCITY <= velocity <= MAX_VELOCITY:
                    # Prefer candidates moving at our expected 10px/frame
                    err = abs(velocity - self.lock['velocity'])
                    if err < min_err:
                        min_err = err
                        best_match = cand
            
            if best_match:
                # Update lock with new position and smooth the velocity
                new_v = self.lock['y_top'] - best_match['y_top']
                self.lock = {
                    'y_top': best_match['y_top'], 
                    'lost_count': 0, 
                    'velocity': (self.lock['velocity'] + new_v) // 2
                }
                return self.lock, intensities
            else:
                # COAST MODE: If no match, predict where it should be
                self.lock['y_top'] -= self.lock['velocity']
                self.lock['lost_count'] += 1
                
                # If it exits the scan zone or stays lost too long, break lock
                if self.lock['y_top'] < SCAN_Y_START or self.lock['lost_count'] > COAST_LIMIT:
                    self.lock = None
                return self.lock, intensities

def draw_pixel_ruler(img):
    for y in range(0, img.shape[0], 50):
        cv2.line(img, (0, y), (15, y), (200, 200, 200), 1)
        cv2.putText(img, str(y), (20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return img

def draw_telemetry(img, intensities, lock):
    h, w, _ = img.shape
    gx = w - 120
    cv2.rectangle(img, (gx, 0), (w, h), (20, 20, 20), -1)
    for y in range(1, h):
        x1 = int(gx + (intensities[y-1]/100)*100)
        x2 = int(gx + (intensities[y]/100)*100)
        cv2.line(img, (x1, y-1), (x2, y), (0, 255, 0), 1)
    if lock:
        cv2.rectangle(img, (gx, lock['y_top']), (w, lock['y_top']+BANNER_H), (0, 0, 255), 2)
    return img

def run_sentinel_theta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170

    ts = TrajectorySentinel()
    manifest = []
    
    # Analyze the window frame-by-frame
    for i in range(target_idx - 20, target_idx + 80):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lock, intensities = ts.update(img_gray)
        
        overlay = img_bgr.copy()
        if lock:
            # Check if this is a "Coasted" frame or a "Real" frame
            color = (0, 255, 255) if lock['lost_count'] > 0 else (0, 0, 255)
            cv2.rectangle(overlay, (0, lock['y_top']), (img_bgr.shape[1], lock['y_top']+BANNER_H), color, -1)
            manifest.append({"idx": i, "y_top": lock['y_top'], "y_bot": lock['y_top']+BANNER_H, "coasted": lock['lost_count']})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Add Search Limit Lines
        cv2.line(img_bgr, (0, SCAN_Y_START), (img_bgr.shape[1], SCAN_Y_START), (255, 128, 0), 1)
        cv2.line(img_bgr, (0, SCAN_Y_END), (img_bgr.shape[1], SCAN_Y_END), (255, 128, 0), 1)
        
        img_bgr = draw_pixel_ruler(img_bgr)
        img_bgr = draw_telemetry(img_bgr, intensities, lock)
        
        status = "TRACKING" if lock and lock['lost_count'] == 0 else "COASTING" if lock else "SEARCHING"
        cv2.putText(img_bgr, f"THETA F:{i} | {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"theta_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_theta_manifest.csv", index=False)
    print("Sentinel Theta complete. Path manifest saved.")

if __name__ == "__main__":
    run_sentinel_theta()