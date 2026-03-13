import cv2
import numpy as np
import os

# --- CONFIG ---
TEMPLATE_DIR = "templates"

def get_audit_mask():
    """Matches the 'Surgical Mask' used in the primary detector."""
    mask = np.zeros((48, 48), dtype=np.uint8)
    # Focus on the 'Body' of the ore, ignoring the top HUD text area (0-18)
    # and avoiding the extreme edges where gravel dominates.
    cv2.rectangle(mask, (8, 19), (40, 45), 255, -1)
    return mask

def analyze_template_integrity_v51():
    mask = get_audit_mask()
    
    # 1. Group Templates
    raw_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')]
    groups = {}
    for f in raw_files:
        if any(x in f for x in ["background", "negative"]): continue
        ore_type = f.split("_")[0]
        img = cv2.imread(os.path.join(TEMPLATE_DIR, f), 0)
        if img is not None:
            if ore_type not in groups: groups[ore_type] = []
            groups[ore_type].append({'name': f, 'img': cv2.resize(img, (48, 48))})

    ore_types = sorted(groups.keys())
    print(f"--- Masked Template Integrity Audit (v5.1) ---")
    print(f"[!] Using Surgical Mask (Ignoring HUD & Corners)")

    # 2. CROSS-CLASS DISTINCTNESS
    print("\n[1] Checking Masked Cross-Class Distinctness...")
    for i, type_a in enumerate(ore_types):
        for j, type_b in enumerate(ore_types):
            if i >= j: continue
            
            max_collision = 0
            for t_a in groups[type_a]:
                for t_b in groups[type_b]:
                    # The 'mask' argument here is critical!
                    res = cv2.matchTemplate(t_a['img'], t_b['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    max_collision = max(max_collision, res.max())
            
            # Revised thresholds for masked matching
            if max_collision > 0.82:
                status = "!! COLLISION !!" if max_collision > 0.88 else "WARNING"
                print(f"  > {type_a} vs {type_b}: {max_collision:.3f} [{status}]")

    # 3. INTRA-CLASS DIVERSITY
    print("\n[2] Checking Masked Intra-Class Cohesion...")
    for ore_type in ore_types:
        t_list = groups[ore_type]
        if len(t_list) < 2: continue
            
        scores = []
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                res = cv2.matchTemplate(t_list[i]['img'], t_list[j]['img'], cv2.TM_CCORR_NORMED, mask=mask)
                scores.append(res.max())
        
        avg_cohesion = np.mean(scores)
        status = "HEALTHY" if avg_cohesion > 0.85 else "DIVERSE"
        print(f"  > {ore_type:5}: Cohesion {avg_cohesion:.3f} [{status}]")

if __name__ == "__main__":
    analyze_template_integrity_v51()