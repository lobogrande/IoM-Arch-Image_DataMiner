import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_alpha_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
HUD_DANGER_ZONE = 150 # Velocity-only mode below this line

# KINEMATIC LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 10 
MIN_V, MAX_V = 7.0, 15.0
NUCLEATION_ZONE = 250
# Banners are wide; damage numbers/sprites are narrow.
MIN_HORIZONTAL_WIDTH = 600 

class SiblingClusterAlpha:
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
        """Internal logger for the history buffer."""
        for t in self.tops:
            self.history.append({
                "frame": f, "id": self.id, "y": t, "v": self.v, "valid": self.is_validated
            })

    def update(self, new_tops, f):
        self.age += 1
        
        # --- LAW OF INERTIA: THE HUD BYPASS ---
        # If validated and high on the screen, ignore visual noise to prevent Stage HUD anchoring.
        is_blind = self.is_validated and (self.tops[0] < HUD_DANGER_ZONE)
        
        if is_blind:
            # Pure Momentum Mode: Subtract last known good velocity from current Y
            self.tops = [t - self.v for t in self.tops]
        else:
            # NORMAL TRACKING MODE
            target = self.tops[0] - self.v
            matches = [t for t in new_tops if abs(t - target) < 30]
            
            # Filter matches to ensure they imply UPWARD movement
            matches = [t for t in matches if (self.tops[0] - t) > 1.0]
            
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                
                # Check for consistent "Banner-like" upward velocity.
                # Allow slower separation speeds during early nucleation (age < 15).
                v_min = 2.0 if self.age < 15 else MIN_V
                if v_min <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    # Smooth the velocity vector
                    self.v = (self.v * 0.7) + (actual_v * 0.3)
                else:
                    self.consistency_score = max(0, self.consistency_score - 1)
                
                # Maintain relative spacing for multi-banner clusters
                self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
            else:
                # Predictive Coast: If the signal is lost, use the last velocity
                self.tops = [t - self.v for t in self.tops]
                self.consistency_score = max(0, self.consistency_score - 1)

        # Validation logic (Backtracking Trigger)
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # Execution logic (Kill candidates that stop moving or leave the screen)
        if self.age > 30 and self.consistency_score < 3 and not is_blind: 
            self.active = False
        if self.tops[0] < -50: 
            self.active = False
            
        return self.active

class SentinelAlpha:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_banner_structure(self, img_gray, t, age):
        """Verifies horizontal contiguity using Morphological Closing to bridge text gaps."""
        h, w = img_gray.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        
        # 1. Darkness Mask (Looking for the black rectangle)
        row = img_gray[y_probe, :].reshape(1, -1)
        mask = (row < 75).astype(np.uint8) * 255 # Slightly relaxed intensity
        
        # 2. Horizontal Closing: Bridges the white text gaps
        kernel = np.ones((1, 100), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Measurement: Find the widest continuous dark segment
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        if num_labels <= 1: return False
        
        max_width = np.max(stats[1:, cv2.CC_STAT_WIDTH])
        
        # Width Requirement: Starts small at nucleation and scales up to 600px
        # This captures the 'narrow' banners at frame 2013 and 2886
        target_width = MIN_HORIZONTAL_WIDTH if age > 15 else (w * 0.20)
        return max_width > target_width

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            
            # Find age if this top is already tracked
            age = 0
            for c in self.clusters:
                if any(abs(t - ct) < 30 for ct in c.tops):
                    age = c.age
                    break
            
            if self.check_banner_structure(img_gray, t, age):
                valid_tops.append(t)
        
        # 1. Update and Backfill
        for c in self.clusters:
            was_validated = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_validated:
                    self.final_manifest.extend(c.history) # Dump backfilled history
                elif c.is_validated:
                    self.final_manifest.append(c.history[-1]) # Add latest frame
        
        # 2. Nucleation logic (Birth zone only)
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 50 for c in self.clusters for t in c.tops)]
        if birth:
            self.clusters.append(SiblingClusterAlpha(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_alpha():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    sa = SentinelAlpha()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        clusters = sa.process_frame(img_gray, i)
        
        # Visual Debug Overlay
        overlay = img_bgr.copy()
        for c in clusters:
            color = (0, 0, 255) if c.is_validated else (0, 255, 255) # Red = Valid, Yellow = Candidate
            for t in c.tops:
                y_draw = int(t + DRAW_OFFSET)
                cv2.rectangle(overlay, (0, max(0, y_draw)), 
                              (img_bgr.shape[1], max(0, y_draw + BANNER_H)), color, -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        status = f"TRACKS: {len(clusters)} | VALID: {len([c for c in clusters if c.is_validated])}"
        cv2.putText(img_bgr, f"ALPHA F:{i} | {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"alpha_{i:05}.png"), img_bgr)
    
    # Save the backfilled, frame-perfect manifest
    pd.DataFrame(sa.final_manifest).to_csv("sentinel_alpha_manifest.csv", index=False)
    print("Sentinel Alpha stress test complete.")

if __name__ == "__main__":
    run_sentinel_alpha()