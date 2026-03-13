import cv2
import numpy as np
import os
import pandas as pd # Optional, but great for the summary table

# --- CONFIG ---
TEMPLATE_DIR = "templates"

def analyze_template_integrity():
    # 1. Load and Group Templates by Ore Type
    raw_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')]
    groups = {}
    
    for f in raw_files:
        if any(x in f for x in ["background", "negative"]): continue
        ore_type = f.split("_")[0]
        if ore_type not in groups: groups[ore_type] = []
        
        img = cv2.imread(os.path.join(TEMPLATE_DIR, f), 0)
        if img is not None:
            groups[ore_type].append({'name': f, 'img': cv2.resize(img, (48, 48))})

    ore_types = sorted(groups.keys())
    results = []

    print(f"--- Template Integrity Audit (Found {len(ore_types)} Classes) ---")

    # 2. CROSS-CLASS DISTINCTNESS (Question 1)
    # We compare every class against every other class to find 'Collision Risks'
    print("\n[1] Checking Cross-Class Distinctness (Collision Risks)...")
    for i, type_a in enumerate(ore_types):
        for j, type_b in enumerate(ore_types):
            if i >= j: continue # Avoid double-checking
            
            # Find the highest match between any two templates of different types
            max_collision = 0
            for t_a in groups[type_a]:
                for t_b in groups[type_b]:
                    res = cv2.matchTemplate(t_a['img'], t_b['img'], cv2.TM_CCORR_NORMED)
                    max_collision = max(max_collision, res.max())
            
            status = "!! DANGER !!" if max_collision > 0.90 else "WARNING" if max_collision > 0.85 else "SAFE"
            if max_collision > 0.85:
                print(f"  > {type_a} vs {type_b}: Max Match {max_collision:.3f} [{status}]")

    # 3. INTRA-CLASS DIVERSITY (Question 2)
    # We check if the templates WITHIN a class are consistent
    print("\n[2] Checking Intra-Class Diversity (Consistency)...")
    for ore_type in ore_types:
        t_list = groups[ore_type]
        if len(t_list) < 2:
            print(f"  > {ore_type}: Only 1 template. Cannot assess diversity.")
            continue
            
        # Calculate 'Internal Cohesion' (Average match between templates of same type)
        internal_scores = []
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                res = cv2.matchTemplate(t_list[i]['img'], t_list[j]['img'], cv2.TM_CCORR_NORMED)
                internal_scores.append(res.max())
        
        avg_cohesion = np.mean(internal_scores)
        std_dev = np.std(internal_scores)
        
        # Low Cohesion means the templates look too different (Lighting/Variety problem)
        # High Std Dev means you have 'Outlier' templates that don't match the group
        status = "HEALTHY" if avg_cohesion > 0.90 else "THIN (Needs more samples)"
        print(f"  > {ore_type}: Cohesion {avg_cohesion:.3f} | Variance {std_dev:.4f} [{status}]")

if __name__ == "__main__":
    analyze_template_integrity()