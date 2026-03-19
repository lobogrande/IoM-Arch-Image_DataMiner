import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_tau_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
ICON_SAFETY_MARGIN = 20

# KINEMATIC & SIGNAL LIMITS
EXPECTED_V = 10.0
VALIDATION_DISPLACEMENT = 50.0 
STALL_KILL_LIMIT = 6 
NUCLEATION_ZONE = 300 

class SiblingClusterTau:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.dist_moved = 0.0
        self.stall_count = 0
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        
        # RETROACTIVE BUFFER: Stores frames before validation
        self.history = []
        self._record(frame_idx)

    def _record(self, frame_idx):
        """Internal method to log position to history."""
        for t in self.tops:
            self.history.append({
                "frame": frame_idx, 
                "id": self.id, 
                "y_top": t, 
                "v": self.v,
                "is_validated": self.is_validated
            })

    def update(self, new_tops, frame_idx):
        self.age += 1
        target_leader = self.tops[0] - self.v
        matches = [t for t in new_tops if abs(t - target_leader) < 25]
        
        if matches:
            best_leader = min(matches, key=lambda t: abs(t - target_leader))
            actual_v = self.tops[0] - best_leader
            
            if abs(actual_v) < 1.5:
                self.stall_count += 1
            else:
                self.stall_count = 0
                self.dist_moved += actual_v
            
            self.v = (self.v * 0.7) + (actual_v * 0.3)
            self.tops = [best_leader] + [best_leader + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            self.tops = [t - self.v for t in self.tops]
            self.dist_moved += self.v
            self.stall_count = 0 

        # Validation Check
        if not self.is_validated and self.dist_moved >= VALIDATION_DISPLACEMENT:
            self.is_validated = True
            # Update history items to reflect validation
            for item in self.history: item['is_validated'] = True

        self._record(frame_idx)

        # Kinematic Laws
        if self.stall_count > STALL_KILL_LIMIT: self.active = False
        if (self.tops[0] + BANNER_H + ICON_SAFETY_MARGIN) < SCAN_Y_START: self.active = False
                
        return self.active

class SentinelTau:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def get_structured_edges(self, img_gray):
        h, w = img_gray.shape
        # 5-Point Horizontal check (Width verification)
        cols = [int(w*0.1), int(w*0.3), int(w*0.5), int(w*0.7), int(w*0.9)]
        ints = np.mean(img_gray[:, cols], axis=1)
        
        grad = np.diff(ints.astype(float))
        tops = np.where(grad < -7.0)[0]
        
        valid_tops = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            
            # Internal Row Variance (Text vs Empty Grid)
            if t + 22 < h:
                row_slice = img_gray[t + 22, cols[1]:cols[3]]
                if np.var(row_slice) > 5.0: # Check for text signal
                    if np.mean(ints[t:t+40]) < 48.0:
                        valid_tops.append(t)
        return valid_tops

    def process_frame(self, img_gray, frame_idx):
        tops = self.get_structured_edges(img_gray)
        
        # 1. Update and Check for Validation Trigger
        for c in self.clusters:
            was_validated = c.is_validated
            if c.update(tops, frame_idx):
                # If it JUST became validated, dump its entire history to the manifest
                if c.is_validated and not was_validated:
                    self.final_manifest.extend(c.history)
                elif c.is_validated:
                    # If already validated, just add the latest frame
                    self.final_manifest.append(c.history[-1])
            
        # 2. Nucleation
        birth_tops = [t for t in tops if t > NUCLEATION_ZONE]
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 50 for c in self.clusters for t in c.tops)]
        if birth_tops:
            self.clusters.append(SiblingClusterTau(birth_tops, frame_idx))

        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_tau():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    st = SentinelTau()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        clusters = st.process_frame(img_gray, i)
        
        # Visual Debug: Only draw clusters that are currently validated
        overlay = img_bgr.copy()
        for c in clusters:
            if c.is_validated:
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, y_draw), (img_bgr.shape[1], y_draw + BANNER_H), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        status = f"TRACKS: {len(clusters)} | VALID: {len([c for c in clusters if c.is_validated])}"
        cv2.putText(img_bgr, f"TAU F:{i} | {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"tau_{i:05}.png"), img_bgr)

    pd.DataFrame(st.final_manifest).to_csv("sentinel_tau_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_tau()