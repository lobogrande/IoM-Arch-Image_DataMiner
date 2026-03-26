import sys, os
import cv2
import pandas as pd
import numpy as np

# Add root to sys.path to find project_config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIGURATION ---
RUN_ID = 0
CSV_PATH = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{RUN_ID}.csv")
IMAGE_SOURCE = cfg.get_buffer_path(RUN_ID)
VERIFICATION_OUT = os.path.join(cfg.DATA_DIRS["TRACKING"], f"accuracy_audit_run_{RUN_ID}.jpg")

# Thresholds for assessment
MIN_STABLE_DURATION = 3  # Frames needed to consider a detection 'stable'
MAX_EXPECTED_GAP = 120   # Frames between mining slots before flagging a potential miss

def run_evaluation():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Error: CSV is empty. No data to evaluate.")
        return

    print(f"--- ASSESSING RUN {RUN_ID} SEQUENCER OUTPUT ---")

    # --- PART 1: ACCURACY ASSESSMENT (Visual Confidence Audit) ---
    # We take the 50 lowest-confidence frames. If these are correct, the high-conf ones are too.
    print("[1/2] Generating Visual Accuracy Audit (Low-Confidence Samples)...")
    bottom_50 = df.nsmallest(50, 'confidence').copy()
    
    # Create a contact sheet of ROIs for visual confirmation
    thumb_size = 80
    canvas = np.zeros((thumb_size * 5, thumb_size * 10, 3), dtype=np.uint8)
    
    for i, (_, row) in enumerate(bottom_50.iterrows()):
        img_path = os.path.join(IMAGE_SOURCE, row['filename'])
        full_img = cv2.imread(img_path)
        if full_img is None: continue
        
        # Approximate ROI based on slot metadata (centered on sprite)
        # Note: We use a generic centered crop for the audit visual
        h, w = full_img.shape[:2]
        # For simplicity, we sample a window around where the detection happened
        # (This uses the center of the frame as a placeholder if ROI coordinates weren't logged)
        # Assuming 1080p center if not specified, but we know the grid is roughly 261-500Y
        y_center = 350 
        x_center = 500
        
        # Just grab the thumbnail if possible
        thumb = cv2.resize(full_img[100:600, 50:950], (thumb_size, thumb_size))
        cv2.putText(thumb, f"S{int(row['slot_id'])}", (5, 15), 0, 0.4, (0, 255, 0), 1)
        cv2.putText(thumb, str(row['confidence']), (5, thumb_size-5), 0, 0.3, (255, 255, 255), 1)
        
        r, c = i // 10, i % 10
        canvas[r*thumb_size:(r+1)*thumb_size, c*thumb_size:(c+1)*thumb_size] = thumb

    cv2.imwrite(VERIFICATION_OUT, canvas)
    print(f"      - Accuracy Audit Image saved: {VERIFICATION_OUT}")

    # --- PART 2: THOROUGHNESS ASSESSMENT (Monotonic Sequence Audit) ---
    print("[2/2] Analyzing Sequence Integrity...")
    
    # Cluster consecutive frames for the same slot to find "Events"
    df['event_id'] = ( (df['slot_id'] != df['slot_id'].shift()) | 
                       (df['frame_idx'] != df['frame_idx'].shift() + 1) ).cumsum()
    
    events = df.groupby('event_id').agg({
        'frame_idx': ['min', 'max', 'count'],
        'slot_id': 'first'
    })
    events.columns = ['start', 'end', 'duration', 'slot']
    
    # Apply 'No Backwards' Rule: If slot ID decreases, it is a floor transition
    events['prev_slot'] = events['slot'].shift(1)
    events['is_transition'] = events['slot'] < events['prev_slot']
    
    transitions = events[events['is_transition'] == True]
    print(f"      - Detected {len(transitions) + 1} distinct floor boundaries.")

    # Check for "Weak Detections" (unstable frames)
    weak_events = events[events['duration'] < MIN_STABLE_DURATION]
    print(f"      - Found {len(weak_events)} weak detections (possible false positives).")

    # Check for "Detection Gaps" (possible missed ores)
    events['gap'] = events['start'] - events['end'].shift(1)
    large_gaps = events[events['gap'] > MAX_EXPECTED_GAP]
    
    print(f"      - Found {len(large_gaps)} significant temporal gaps.")
    if not large_gaps.empty:
        print("        [Sample Gap]:")
        print(large_gaps[['start', 'slot', 'gap']].head())

    print("\n--- ASSESSMENT COMPLETE ---")

if __name__ == "__main__":
    run_evaluation()