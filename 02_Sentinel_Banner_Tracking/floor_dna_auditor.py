# floor_dna_auditor.py
# Purpose: De-duplicate sequencer frames using Row 4 "DNA" signatures.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

INPUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
FINAL_OUT = os.path.join(cfg.DATA_DIRS["TRACKING"], "unique_floor_candidates.csv")

# DNA Config: Row 4 center is roughly Y=438
DNA_Y_CENTER = 438
DNA_HEIGHT = 10 

def get_dna_signature(img):
    # Slice Row 4 across the grid area
    dna_strip = img[DNA_Y_CENTER - 5 : DNA_Y_CENTER + 5, 60:850]
    # Simple Pixel Hash: Mean intensity of 20px horizontal buckets
    buckets = np.mean(dna_strip.reshape(-1, 20), axis=0)
    return "-".join([str(round(v, 1)) for v in buckets])

def run_dna_audit():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV)
    
    print(f"--- TIERED DNA AUDIT ---")
    print(f"Analyzing DNA for {len(df)} candidate frames...")
    
    unique_candidates = []
    seen_dna = set()
    
    for _, row in df.iterrows():
        img_path = os.path.join(cfg.get_buffer_path(0), row['filename'])
        img = cv2.imread(img_path, 0)
        if img is None: continue
        
        # Tier 1: Row 4 DNA Signature
        dna = get_dna_signature(img)
        
        # We only keep the EARLIEST frame for each unique DNA signature
        if dna not in seen_dna:
            seen_dna.add(dna)
            row_dict = row.to_dict()
            row_dict['dna_sig'] = dna
            unique_candidates.append(row_dict)

    out_df = pd.DataFrame(unique_candidates)
    out_df.to_csv(FINAL_OUT, index=False)
    print(f"[DONE] Pruned {len(df)} frames down to {len(out_df)} unique floor candidates.")

if __name__ == "__main__":
    run_dna_audit()