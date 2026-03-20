import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import os
import json
import shutil

# --- ALIGNMENT CONFIGURATION ---
UNIFIED_ROOT = "Unified_Consensus_Inputs"
W2_SOURCE = "Final_Consensus_Images"
COLLAPSED_SOURCE = "Consensus_Sentinel_Results"

def align_archaeology_inputs():
    if not os.path.exists(UNIFIED_ROOT): os.makedirs(UNIFIED_ROOT)
    
    # 1. PROCESS DATASET 0 (Source: Worker 2)
    ds0_path = os.path.join(UNIFIED_ROOT, "Run_0")
    if os.path.exists(ds0_path): shutil.rmtree(ds0_path)
    
    w2_run0_img_dir = os.path.join(W2_SOURCE, "Run_0")
    w2_run0_json = "final_sequence_run_0.json"
    
    if os.path.exists(w2_run0_img_dir) and os.path.exists(w2_run0_json):
        print("--- Aligning Run_0 (Worker 2) ---")
        shutil.copytree(w2_run0_img_dir, ds0_path)
        shutil.copy2(w2_run0_json, os.path.join(ds0_path, "final_sequence.json"))
        print(f" [+] Run_0 consolidated from {W2_SOURCE}")
    else:
        print(" [!] Warning: Run_0 source files from Worker 2 not found.")

    # 2. PROCESS DATASETS 1-4 (Source: Collapsed Worker)
    for ds_id in ["1", "2", "3", "4"]:
        unified_ds_path = os.path.join(UNIFIED_ROOT, f"Run_{ds_id}")
        if os.path.exists(unified_ds_path): shutil.rmtree(unified_ds_path)
        
        col_run_img_dir = os.path.join(COLLAPSED_SOURCE, f"Run_{ds_id}")
        col_run_json = f"consensus_sequence_run_{ds_id}.json"
        
        if os.path.exists(col_run_img_dir) and os.path.exists(col_run_json):
            print(f"--- Aligning Run_{ds_id} (Collapsed Worker) ---")
            shutil.copytree(col_run_img_dir, unified_ds_path)
            shutil.copy2(col_run_json, os.path.join(unified_ds_path, "final_sequence.json"))
            print(f" [+] Run_{ds_id} consolidated from {COLLAPSED_SOURCE}")
        else:
            print(f" [!] Warning: Run_{ds_id} source files from Collapsed Worker not found.")

    print(f"\nAlignment Complete. All data unified in: {UNIFIED_ROOT}")

if __name__ == "__main__":
    align_archaeology_inputs()