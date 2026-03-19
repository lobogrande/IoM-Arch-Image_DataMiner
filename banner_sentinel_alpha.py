import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_iota_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# SEARCH AREA (Minigame Zone)
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H_TARGET = 45

# DAMPED MOTION PARAMETERS
GLOBAL_VELOCITY = 10.0 # Our "Target" speed from Ground Truth
VELOCITY_SMOOTHING = 0.95 # 95% Weight on global/previous, 5% on new data
MAX_COAST = 12

class SentinelIota:
    def __init__(self):
        self.lock = None # {'y_top': y, 'v': 10.0, 'lost': 0}
        
    def get_banner_edges(self, intensities):
        """Finds the sharpest 'Down-Up' edge pairs (the valley)."""
        # Calculate the 1D gradient (change in intensity)
        diff = np.diff(intensities.astype(float))
        
        # Candidate top edges (Intensity drops > 5)
        tops = np.where(diff < -5.0)[0]
        # Candidate bottom edges (Intensity rises > 5)
        bots = np.where(diff > 5.0)[0]
        
        candidates = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            for b in bots:
                if b <= t: continue
                height = b - t
                if 35 <= height <= 65: # The vertical height of a banner container
                    # Calculate internal darkness (The floor of the valley)
                    darkness = np.mean(intensities[t+2:b-2])
                    if darkness < 40.0:
                        candidates.append({'y_top': t, 'y_bot': b, 'dark': darkness})
        
        # Sort by darkness (the deepest valley is usually the banner)
        candidates.sort(key=lambda x: x['dark'])
        return candidates

    def update(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        
        candidates = self.get_banner_edges(intensities)
        
        if self.lock is None:
            # SEARCH: Look for a new banner appearing in the grid (Y > 200)
            for cand in candidates:
                if cand['y_top'] > 200:
                    self.lock = {'y_top': cand['y_top'], 'v': GLOBAL_VELOCITY, 'lost': 0}
                    return self.lock, intensities
            return None, intensities
        else:
            # TRACK: Find candidate closest to (Last_Y - Velocity)
            best_cand = None
            min_dist = 30 # Search radius
            
            target_y = self.lock['y_top'] - self.lock['v']
            
            for cand in candidates:
                dist = abs(cand['y_top'] - target_y)
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand
            
            if best_cand:
                # Calculate new instantaneous velocity
                measured_v = self.lock['y_top'] - best_cand['y_top']
                # HEAVY DAMPING: Prevents the "Outrun" bug
                # New velocity is 95% the old/expected speed, 5% the new measurement
                self.lock['v'] = (self.lock['v'] * VELOCITY_SMOOTHING) + (measured_v * (1.0 - VELOCITY_SMOOTHING))
                self.lock['y_top'] = best_cand['y_top']
                self.lock['lost'] = 0
                return self.lock, intensities
            else:
                # COAST: Move at current damped velocity
                self.lock['y_top'] -= self.lock['v']
                self.lock['lost'] += 1
                if self.lock['lost'] > MAX_COAST or self.lock['y_top'] < SCAN_Y_START:
                    self.lock = None
                return self.lock, intensities

def run_sentinel_iota():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170

    si = SentinelIota()
    manifest = []
    
    for i in range(target_idx - 20, target_idx + 80):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lock, intensities = si.update(img_gray)
        
        overlay = img_bgr.copy()
        if lock:
            color = (0, 255, 255) if lock['lost'] > 0 else (0, 0, 255)
            cv2.rectangle(overlay, (0, int(lock['y_top'])), (img_bgr.shape[1], int(lock['y_top']+BANNER_H_TARGET)), color, -1)
            manifest.append({"idx": i, "y_top": lock['y_top'], "coasted": lock['lost']})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Telemetry Graph
        gx = img_bgr.shape[1] - 120
        cv2.rectangle(img_bgr, (gx, 0), (img_bgr.shape[1], img_bgr.shape[0]), (20, 20, 20), -1)
        for y in range(1, img_bgr.shape[0]):
            x1 = int(gx + (intensities[y-1]/100)*100)
            x2 = int(gx + (intensities[y]/100)*100)
            cv2.line(img_bgr, (x1, y-1), (x2, y), (0, 255, 0), 1)
        
        status = f"V:{si.lock['v']:.1f}" if si.lock else "SEARCH"
        cv2.putText(img_bgr, f"IOTA F:{i} | {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"iota_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_iota_manifest.csv", index=False)
    print("Sentinel Iota complete.")

if __name__ == "__main__":
    run_sentinel_iota()