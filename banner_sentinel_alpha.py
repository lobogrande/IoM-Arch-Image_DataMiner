import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_psi_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
HUD_DANGER_ZONE = 140 # Switch to momentum-only mode here to prevent HUD anchoring

# KINEMATIC LAWS
EXPECTED_V = 10.0
CONSISTENCY_WINDOW = 12 
MIN_V, MAX_V = 7.0, 15.0
NUCLEATION_ZONE = 250
MIN_HORIZONTAL_WIDTH = 500 # Banners are wide; damage numbers/sprites are narrow

class SiblingClusterPsi:
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
        """Logs current state into the internal history buffer."""
        for t in self.tops:
            self.history.append({
                "frame": f, 
                "id": self.id, 
                "y": t, 
                "v": self.v, 
                "valid": self.is_validated
            })

    def update(self, new_tops, f):
        self.age += 1
        
        # --- LAW 1: INERTIAL BLINDFOLD ---
        # If we are validated and high on the screen, ignore visual matches 
        # to avoid the tracker snapping to the stationary Stage HUD.
        is_blind = self.is_validated and (self.tops[0] < HUD_DANGER_ZONE)
        
        if is_blind:
            # PURE MOMENTUM: No visual matching allowed.
            self.tops = [t - self.v for t in self.tops]
        else:
            # NORMAL TRACKING
            target = self.tops[0] - self.v
            matches = [t for t in new_tops if abs(t - target) < 30]
            
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                
                # Check for "Banner-like" upward velocity.
                # We relax this slightly during early nucleation (age < 12).
                v_min = 2.0 if self.age < 12 else MIN_V
                if v_min <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    # Smoothly update velocity based on actual movement
                    self.v = (self.v * 0.7) + (actual_v * 0.3)
                else:
                    self.consistency_score = max(0, self.consistency_score - 1)
                
                # Snap the cluster to the new match, maintaining relative sibling spacing
                self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
            else:
                # COASTING: Maintain current trajectory when signal is obscured
                self.tops = [t - self.v for t in self.tops]
                self.consistency_score = max(0, self.consistency_score - 1)

        # Validation Logic
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            # Backfill the 'valid' status for the history buffer
            for item in self.history: 
                item['valid'] = True

        self._record(f)

        # Execution Logic (Cleaning up failed candidates or expired tracks)
        if self.age > 20 and self.consistency_score < 3 and not is_blind: 
            self.active = False
        if self.tops[0] < -20: # Full screen exit
            self.active = False
            
        return self.active

class SentinelPsi:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_horizontal_contiguity(self, img_gray, t):
        """Measures the width of the dark bar to filter damage numbers and sprites."""
        h, w = img_gray.shape
        # Probe the row slightly below the detected edge
        y_probe = int(t + 10)
        if y_probe >= h: return False
        
        row = img_gray[y_probe, :]
        dark_pixels = (row < 55).astype(np.uint8)
        
        # Identify the largest contiguous dark segment in this row
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_pixels.reshape(1, -1), connectivity=8
        )
        if num_labels <= 1: return False
        
        max_width = np.max(stats[1:, cv2.CC_STAT_WIDTH])
        return max_width > MIN_HORIZONTAL_WIDTH

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        # Use a center-strip for edge detection to avoid UI edges
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        # Filter edges for both darkness and horizontal width
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END 
                      and self.check_horizontal_contiguity(img_gray, t)]
        
        # 1. Update existing tracks
        for c in self.clusters:
            was_validated = c.is_validated
            if c.update(valid_tops, f_idx):
                # If it JUST passed validation, dump the backfilled history to manifest
                if c.is_validated and not was_validated:
                    self.final_manifest.extend(c.history)
                elif c.is_validated:
                    # Otherwise just add the latest frame
                    self.final_manifest.append(c.history[-1])
        
        # 2. Nucleation (New banner arrival)
        birth_tops = [t for t in valid_tops if t > NUCLEATION_ZONE]
        # Prevent spawning a duplicate tracker on an already tracked banner
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 40 for c in self.clusters for t in c.tops)]
        
        if birth_tops:
            self.clusters.append(SiblingClusterPsi(birth_tops, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_psi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    sp = SentinelPsi()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        clusters = sp.process_frame(img_gray, i)
        
        # Visual Debug Output
        overlay = img_bgr.copy()
        for c in clusters:
            # Color coding: Yellow = Candidate, Red = Validated Banner
            color = (0, 0, 255) if c.is_validated else (0, 255, 255)
            for t in c.tops:
                y_draw = int(t + DRAW_OFFSET)
                cv2.rectangle(overlay, (0, max(0, y_draw)), 
                              (img_bgr.shape[1], max(0, y_draw + BANNER_H)), color, -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        status = f"TRACKS: {len(clusters)} | VALID: {len([c for c in clusters if c.is_validated])}"
        cv2.putText(img_bgr, f"PSI F:{i} | {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"psi_{i:05}.png"), img_bgr)
    
    # Save the final validated manifest
    pd.DataFrame(sp.final_manifest).to_csv("sentinel_psi_manifest.csv", index=False)
    print("Sentinel Psi run complete.")

if __name__ == "__main__":
    run_sentinel_psi()