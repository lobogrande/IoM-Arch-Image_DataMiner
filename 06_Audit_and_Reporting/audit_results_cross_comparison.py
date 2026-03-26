import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import csv
import os
import sys

# --- 1. DIRECTORY CONFIGURATION ---
# Update these to match your actual folder names!
CENSUS_CSV = "high_fidelity_v30_2_results.csv"    # Your hand-checked 'Gold Standard'
CENSUS_IMG_DIR = "audit_verification" # Where your manual images live

FORENSIC_CSV = "final_perfect_audit_v31_15.csv" # The new automated v31.28+ results
FORENSIC_IMG_DIR = "forensic_verification_endless" # Where the new HUD images are

REPORT_OUTPUT = "DEEP_AUDIT_REPORT.csv"

def load_and_sanitize_data(filepath):
    """
    Reads CSV and ensures Floor/Slot are integers for perfect matching.
   
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"!!! ERROR: Could not find {filepath}")
        return data

    with open(filepath, 'r') as f:
        # We skip the header manually if needed, or use DictReader
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Sanitize: strip whitespace and force to int
                f_num = int(str(row['Floor']).strip())
                slot = int(str(row['Slot']).strip())
                
                if f_num not in data:
                    data[f_num] = {}
                
                data[f_num][slot] = {
                    'tier': row['Tier'].strip(),
                    'score': float(row['Score']),
                    'frame': row['Frame'].strip()
                }
            except Exception as e:
                continue # Skip malformed rows
    return data

def run_folder_aware_validation():
    print("--- STARTING FOLDER-AWARE DEEP AUDIT ---")
    
    census = load_and_sanitize_data(CENSUS_CSV)
    forensic = load_and_sanitize_data(FORENSIC_CSV)

    # Find the floors that exist in BOTH files
    shared_floors = sorted(list(set(census.keys()) & set(forensic.keys())))
    
    if not shared_floors:
        print("!!! WARNING: No overlapping floors found. Check your Floor numbers in the CSVs.")
        # Print a sample to help debug
        print(f"Census Floors Sample: {list(census.keys())[:5]}")
        print(f"Forensic Floors Sample: {list(forensic.keys())[:5]}")
        return

    with open(REPORT_OUTPUT, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Floor", "Status", "Discrepancies", "Avg_Score_Gain", 
            "Census_Img_Path", "Forensic_Img_Path", "Note"
        ])

        for f in shared_floors:
            mismatch_count = 0
            score_diffs = []
            
            for slot in range(24):
                c_data = census[f][slot]
                f_data = forensic[f][slot]
                
                if c_data['tier'] != f_data['tier']:
                    mismatch_count += 1
                score_diffs.append(f_data['score'] - c_data['score'])

            avg_gain = sum(score_diffs) / 24
            
            # Construct actual paths to the images for easy clicking
            # Assumes the 'Frame' in CSV matches the filename in the folder
            c_img = os.path.join(CENSUS_IMG_DIR, census[f][0]['frame'])
            f_img = os.path.join(FORENSIC_IMG_DIR, f"Floor_{f}_Verified_HUD.png")

            status = "STABLE" if mismatch_count == 0 else "CONFLICT"
            note = ""
            if avg_gain > 0.05:
                note = "Significant Quality Improvement in Forensic Scan"

            writer.writerow([f, status, mismatch_count, round(avg_gain, 3), c_img, f_img, note])

    print(f"--- Deep Audit Complete! Report saved to: {REPORT_OUTPUT} ---")

if __name__ == "__main__":
    run_folder_aware_validation()