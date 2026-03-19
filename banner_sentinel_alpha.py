import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_kappa_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# SEARCH AREA
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H_TARGET = 45 # Height of the container

# DAMPED MOTION PARAMETERS
GLOBAL_VELOCITY = 10.0 
VELOCITY_SMOOTHING = 0.95 
MAX_COAST = 15 # Slightly extended to ensure full exit coverage

class SentinelKappa:
    def __init__(self):
        self.lock = None # {'y_top': y, 'v': 10.0, 'lost': 0}
        
    def get_banner_edges(self, intensities):
        """Finds the sharpest 'Down-Up' edge pairs (the valley)."""
        diff = np.diff(intensities.astype(float))
        tops = np.where(diff < -5.0)[0]
        bots = np.where(diff > 5.0)[0]
        
        candidates = []
        for t in tops:
            # We allow the search to find the top edge as long as it's within scan bounds
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            for b in bots:
                if b <= t: continue
                height = b - t
                if 35 <= height <= 65:
                    darkness = np.mean(intensities[t+2:b-2])
                    if darkness < 40.0:
                        candidates.append({'y_top': t, 'y_bot': b, 'dark': darkness})
        
        candidates.sort(key=lambda x: x['dark'])
        return candidates

    def update(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        
        candidates = self.get_banner_edges(intensities)
        
        if self.lock is None:
            for cand in candidates:
                if cand['y_top'] > 200:
                    self.lock = {'y_top': float(cand['y_top']), 'v': GLOBAL_VELOCITY, 'lost': 0}
                    return self.lock, intensities
            return None, intensities
        else:
            best_cand = None
            min_dist = 30 
            target_y = self.lock['y_top'] - self.lock['v']
            
            for cand in candidates:
                dist = abs(cand['y_top'] - target_y)
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand
            
            if best_cand:
                measured_v = self.lock['y_top'] - best_cand['y_top']
                self.lock['v'] = (self.lock['v'] * VELOCITY_SMOOTHING) + (measured_v * (1.0 - VELOCITY_SMOOTHING))
                self.lock['y_top'] = float(best_cand['y_top'])
                self.lock['lost'] = 0
            else:
                # COAST: Maintain trajectory
                self.lock['y_top'] -= self.lock['v']
                self.lock['lost'] += 1
            
            # --- UPDATED EXIT CONDITION ---
            # We track until the BOTTOM of the banner clears the SCAN_Y_START
            y_bottom = self.lock['y_top'] + BANNER_H_TARGET
            if y_bottom < SCAN_Y_START or self.lock['lost'] > MAX_COAST:
                self.lock = None
                
            return self.lock, intensities

def draw_pixel_ruler(img):
    for y in range(0, img.shape[0], 50):
        cv2.line(img, (0, y), (15, y), (200, 200, 200), 1)
        cv2.putText(img, str(y), (20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return img

def run_sentinel_kappa():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170

    sk = SentinelKappa()
    manifest = []
    
    # Range extended to ensure we capture the final exit (748+)
    for i in range(target_idx - 20, target_idx + 85):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lock, intensities = sk.update(img_gray)
        
        overlay = img_bgr.copy()
        if lock:
            y_top_int = int(lock['y_top'])
            # Draw the mask (clipped to image boundaries)
            y1, y2 = max(0, y_top_int), max(0, y_top_int + BANNER_H_TARGET)
            cv2.rectangle(overlay, (0, y1), (img_bgr.shape[1], y2), (0, 0, 255), -1)
            manifest.append({"idx": i, "y_top": lock['y_top'], "coasted": lock['lost']})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Telemetry and Rulers
        gx = img_bgr.shape[1] - 120
        cv2.rectangle(img_bgr, (gx, 0), (img_bgr.shape[1], img_bgr.shape[0]), (20, 20, 20), -1)
        for y in range(1, img_bgr.shape[0]):
            x1 = int(gx + (intensities[y-1]/100)*100)
            x2 = int(gx + (intensities[y]/100)*100)
            cv2.line(img_bgr, (x1, y-1), (x2, y), (0, 255, 0), 1)
        
        img_bgr = draw_pixel_ruler(img_bgr)
        status = f"V:{sk.lock['v']:.1f} LOST:{sk.lock['lost']}" if sk.lock else "SEARCHING"
        cv2.putText(img_bgr, f"KAPPA F:{i} | {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"kappa_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_kappa_manifest.csv", index=False)
    print("Sentinel Kappa complete. Exit frames 745-748 tracked.")

if __name__ == "__main__":
    run_sentinel_kappa()