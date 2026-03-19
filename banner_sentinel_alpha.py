import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_chi_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
HUD_DANGER_ZONE = 140 # Y-coordinate where signal interference is high

# KINEMATIC LAWS
EXPECTED_V = 10.0
CONSISTENCY_WINDOW = 8 
MIN_V, MAX_V = 7.0, 14.0 # Slightly relaxed for nucleation
NUCLEATION_ZONE = 250

class SiblingClusterChi:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self._record(frame_idx)

    def _record(self, f):
        for t in self.tops:
            self.history.append({"frame": f, "id": self.id, "y": t, "v": self.v, "valid": self.is_validated})

    def update(self, new_tops, f):
        self.age += 1
        target = self.tops[0] - self.v
        
        matches = [t for t in new_tops if abs(t - target) < 30]
        
        if matches:
            best = min(matches, key=lambda t: abs(t - target))
            actual_v = self.tops[0] - best
            
            if MIN_V <= actual_v <= MAX_V:
                self.consistency_score += 1
                self.v = (self.v * 0.7) + (actual_v * 0.3)
            else:
                self.consistency_score = max(0, self.consistency_score - 1)
            
            self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
            self.lost_frames = 0
        else:
            # --- BLIND COASTING ---
            # If validated and in HUD zone, we "Trust the Physics"
            if self.is_validated and self.tops[0] < HUD_DANGER_ZONE:
                self.tops = [t - self.v for t in self.tops]
                self.lost_frames = 0 # Don't count as lost over HUD
            else:
                self.tops = [t - self.v for t in self.tops]
                self.consistency_score = max(0, self.consistency_score - 2)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # KILL logic
        if self.age > 15 and self.consistency_score < 2: self.active = False
        if self.tops[0] < SCAN_Y_START - 40: self.active = False
        return self.active

class SentinelChi:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_adaptive_integrity(self, img_gray, t, age):
        """Uses a growing horizontal window based on banner age."""
        h, w = img_gray.shape
        # Dynamic Aperture: Starts at 30% center, grows to 90%
        aperture = min(0.9, 0.3 + (age * 0.05))
        c1, c2 = int(w * (0.5 - aperture/2)), int(w * (0.5 + aperture/2))
        
        # 1. Darkness Check
        row_strip = img_gray[t + 5, c1:c2]
        if np.mean(row_strip) > 65: return False
        
        # 2. Texture Check (Edge Counting)
        # Instead of max-min, we count pixel transitions to find 'Text'
        row_core = img_gray[t + 22, c1:c2].astype(float)
        diffs = np.abs(np.diff(row_core))
        # A real text bar has many sharp transitions
        if np.sum(diffs > 40) < (10 * aperture): return False
        
        return True

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        # Note: We pass '0' for age to new detections, 
        # but in a real loop we'd check against existing cluster ages.
        valid_tops = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            # Find age if this top is near an existing cluster
            age = 0
            for c in self.clusters:
                if any(abs(t - ct) < 20 for ct in c.tops):
                    age = c.age
                    break
            
            if self.check_adaptive_integrity(img_gray, t, age):
                valid_tops.append(t)
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Birth logic
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 40 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterChi(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_chi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sc = SentinelChi()
    
    for i in range(START_F, min(END_F, len(all_files