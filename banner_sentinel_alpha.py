import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_eta_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- GEOMETRIC CONSTRAINTS ---
SCAN_Y_START = 40   
SCAN_Y_END = 450     

# --- DYNAMIC THRESHOLDS ---
# We use Hysteresis: 
# A "Seed" must be very dark to START a banner.
# A "Bridge" can be lighter to KEEP the banner connected.
SEED_THRESH = 18.0    
BRIDGE_THRESH = 32.0  
MIN_BANNER_H = 20     
MAX_BANNER_H = 95
EXPECTED_VELOCITY = 10 # Pixels/frame

class BannerTracker:
    def __init__(self, top, bot):
        self.top = top
        self.bot = bot
        self.center = (top + bot) // 2
        self.frames_active = 1
        self.velocity = 0
        self.lost_frames = 0

class SentinelEta:
    def __init__(self):
        self.active_trackers = []
        self.stationary_blacklist = [] # List of Y-centers that don't move

    def process_frame(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        
        # 1. GENERATE HYSTERESIS MASK
        # Seeds are the dark "cores" of the banner
        seeds = (intensities < SEED_THRESH).astype(np.uint8)
        # Bridges are the rows that can contain text/icons
        bridges = (intensities < BRIDGE_THRESH).astype(np.uint8)
        
        # Use Morphological Reconstruction logic:
        # We only keep bridge segments that contain at least one seed.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bridges, connectivity=8)
        
        candidates = []
        for i in range(1, num_labels):
            y_start = stats[i, cv2.CC_STAT_TOP]
            y_end = y_start + stats[i, cv2.CC_STAT_HEIGHT]
            
            # Constraint: Must be in scan zone and have a seed row inside it
            if SCAN_Y_START <= y_start and y_end <= SCAN_Y_END:
                if np.any(seeds[y_start:y_end] == 1):
                    if MIN_BANNER_H <= (y_end - y_start) <= MAX_BANNER_H:
                        candidates.append({'top': y_start, 'bot': y_end, 'center': (y_start+y_end)//2})

        # 2. TEMPORAL ASSOCIATION (The "Lock")
        confirmed = []
        new_trackers = []
        
        for cand in candidates:
            matched = False
            for trk in self.active_trackers:
                # Predictive window: Where we expect the banner to be based on velocity
                expected_y = trk.center - (trk.velocity if trk.velocity > 0 else EXPECTED_VELOCITY)
                if abs(cand['center'] - expected_y) < 25:
                    # Update Tracker
                    old_center = trk.center
                    trk.top, trk.bot = cand['top'], cand['bot']
                    trk.center = cand['center']
                    trk.velocity = old_center - trk.center
                    trk.frames_active += 1
                    trk.lost_frames = 0
                    
                    # Stationary Check: If it hasn't moved 5 pixels in 5 frames, it's UI.
                    if trk.frames_active > 5 and abs(trk.velocity) < 1:
                        if trk.center not in self.stationary_blacklist:
                            self.stationary_blacklist.append(trk.center)
                    
                    if trk.center not in self.stationary_blacklist:
                        new_trackers.append(trk)
                        confirmed.append({'top': trk.top, 'bot': trk.bot, 'id': id(trk)})
                    matched = True
                    break
            
            if not matched:
                # Start a new potential tracker
                if not any(abs(cand['center'] - b_y) < 10 for b_y in self.stationary_blacklist):
                    new_trackers.append(BannerTracker(cand['top'], cand['bot']))

        self.active_trackers = new_trackers
        return confirmed, intensities

def run_sentinel_eta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170

    eta = SentinelEta()
    manifest = []
    
    for i in range(target_idx - 20, target_idx + 60):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        zones, intensities = eta.process_frame(img_gray)
        
        overlay = img_bgr.copy()
        for z in zones:
            # All confirmed moving banners are Red
            cv2.rectangle(overlay, (0, z['top']), (img_bgr.shape[1], z['bot']), (0, 0, 255), -1)
            manifest.append({"idx": i, "y_top": z['top'], "y_bot": z['bot']})
            
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Ruler and Telemetry Drawing
        for y in range(0, img_bgr.shape[0], 50):
            cv2.line(img_bgr, (0, y), (15, y), (200, 200, 200), 1)
            cv2.putText(img_bgr, str(y), (20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        cv2.putText(img_bgr, f"ETA F:{i} | BANNERS: {len(zones)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"eta_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_eta_manifest.csv", index=False)
    print(f"Sentinel Eta complete. Stationary blacklist: {eta.stationary_blacklist}")

if __name__ == "__main__":
    run_sentinel_eta()