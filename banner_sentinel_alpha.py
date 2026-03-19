import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H_TARGET = 45
GLOBAL_VELOCITY = 10.0
VELOCITY_SMOOTHING = 0.95
MAX_COAST = 15

class MultiBannerTracker:
    def __init__(self, y_top):
        self.y_top = float(y_top)
        self.v = GLOBAL_VELOCITY
        self.lost = 0
        self.active = True

class SentinelKappaMulti:
    def __init__(self):
        self.trackers = []
        
    def get_candidates(self, intensities):
        diff = np.diff(intensities.astype(float))
        tops = np.where(diff < -5.0)[0]
        bots = np.where(diff > 5.0)[0]
        
        candidates = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            for b in bots:
                if b <= t: continue
                height = b - t
                if 35 <= height <= 65:
                    darkness = np.mean(intensities[t+2:b-2])
                    if darkness < 40.0:
                        candidates.append({'y_top': t, 'y_bot': b})
        return candidates

    def update(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        
        candidates = self.get_candidates(intensities)
        
        # 1. Update existing trackers
        for trk in self.trackers:
            target_y = trk.y_top - trk.v
            best_match = None
            min_dist = 30
            
            for i, cand in enumerate(candidates):
                dist = abs(cand['y_top'] - target_y)
                if dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                cand = candidates.pop(best_match)
                measured_v = trk.y_top - cand['y_top']
                trk.v = (trk.v * VELOCITY_SMOOTHING) + (measured_v * (1.0 - VELOCITY_SMOOTHING))
                trk.y_top = float(cand['y_top'])
                trk.lost = 0
            else:
                trk.y_top -= trk.v
                trk.lost += 1
            
            # Exit check
            if (trk.y_top + BANNER_H_TARGET) < SCAN_Y_START or trk.lost > MAX_COAST:
                trk.active = False

        # 2. Spawn new trackers for remaining candidates
        for cand in candidates:
            # Only spawn if it's in the nucleation zone (Y > 250) 
            # and not already being tracked
            if cand['y_top'] > 250:
                if not any(abs(t.y_top - cand['y_top']) < 40 for t in self.trackers):
                    self.trackers.append(MultiBannerTracker(cand['y_top']))

        # Cleanup inactive
        self.trackers = [t for t in self.trackers if t.active]
        return self.trackers

def run_global_audit():
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sk = SentinelKappaMulti()
    
    manifest = []
    total_frames = len(all_files)
    
    # Pre-allocate Waterfall Map (Height x Width)
    # Using 1080 for height to see full context
    waterfall = np.zeros((600, total_frames), dtype=np.uint8)

    print(f"Starting Global Audit of {total_frames} frames...")
    
    for i, fname in enumerate(all_files):
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, fname), cv2.IMREAD_GRAYSCALE)
        if img_gray is None: continue
        
        active_tracks = sk.update(img_gray)
        
        for trk in active_tracks:
            # Mark the waterfall map
            y_center = int(trk.y_top + (BANNER_H_TARGET/2))
            if 0 <= y_center < 600:
                # Value 255 for real detection, 128 for coasting
                val = 255 if trk.lost == 0 else 128
                waterfall[y_center, i] = val
            
            manifest.append({
                "frame": i,
                "y_top": trk.y_top,
                "velocity": trk.v,
                "is_coasting": trk.lost > 0
            })
            
        if i % 1000 == 0:
            print(f"Processed {i}/{total_frames} frames...")

    # --- SAVE OUTPUTS ---
    pd.DataFrame(manifest).to_csv("global_audit_manifest.csv", index=False)
    
    # Save the Waterfall Plot
    plt.figure(figsize=(20, 10))
    plt.imshow(waterfall, aspect='auto', cmap='hot')
    plt.title('Global Spatio-Temporal Banner Map (Waterfall)')
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Y-Coordinate (Space)')
    plt.colorbar(label='Detection Confidence (White=Solid, Red=Coasting)')
    plt.savefig('global_banner_waterfall.png', dpi=300)
    
    print("Audit Complete. See 'global_banner_waterfall.png' for the bird's eye view.")

if __name__ == "__main__":
    run_global_audit()