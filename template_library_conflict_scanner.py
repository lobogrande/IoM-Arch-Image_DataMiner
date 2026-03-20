import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
# We pick a floor that has many empty slots to use as a 'Noise Baseline'
NOISE_FLOORS = [2, 7] 
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def run_library_conflict_scan():
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)

    # 1. Load ALL Ore Templates
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    
    # Identify known empty slots from your visual review (e.g., F7 has many)
    # Let's check a wide sample of slots across the noise floors
    test_slots = range(24) 
    
    blacklist = {}

    print(f"--- Scanning Library for Run {TARGET_RUN} Noise Conflicts ---")

    for f_num in NOISE_FLOORS:
        files = [f for f in os.listdir(run_path) if f.startswith(f"F{f_num}_")]
        if not files: continue
        gray = cv2.imread(os.path.join(run_path, files[0]), 0)

        for slot in test_slots:
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            roi = gray[cy-24:cy+24, cx-24:cx+24]
            
            # If this slot is actually empty in reality, any high match here is a 'Ghost'
            for t in ore_templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                
                if score > 0.82: # High threshold for 'definitely a ghost'
                    if t['name'] not in blacklist:
                        blacklist[t['name']] = []
                    blacklist[t['name']].append(score)

    # 2. Report Findings
    print(f"\n[FOUND {len(blacklist)} CONFLICTING TEMPLATES]")
    sorted_blacklist = sorted(blacklist.items(), key=lambda x: max(x[1]), reverse=True)
    
    with open("template_blacklist_report.txt", "w") as f:
        f.write("Template Name | Max Conflict Score | Occurrences\n")
        f.write("-" * 60 + "\n")
        for name, scores in sorted_blacklist:
            line = f"{name:<35} | {max(scores):.4f} | {len(scores)}"
            print(line)
            f.write(line + "\n")

    print("\nResults saved to template_blacklist_report.txt")
    print("These templates are 'synonyms' for floor noise and should be moved to a quarantine folder.")

if __name__ == "__main__":
    run_library_conflict_scan()