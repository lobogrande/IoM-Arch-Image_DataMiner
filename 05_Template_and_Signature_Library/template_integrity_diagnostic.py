import cv2
import numpy as np
import os

# --- CONFIG ---
TEMPLATE_DIR = "templates"

def get_feature_mask():
    mask = np.zeros((48, 48), dtype=np.uint8)
    # Surgical center-focus (Adjusted to 24x26 heart of the ore)
    cv2.rectangle(mask, (12, 20), (36, 44), 255, -1)
    return mask

def process_for_features(img):
    """Applies edge detection to emphasize gem shapes over rock texture."""
    # 1. Blur to remove noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # 2. Sobel Edge Detection (Vertical + Horizontal)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    # 3. Combine and convert back to uint8
    combined = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.clip(combined, 0, 255))

def analyze_template_integrity_v52():
    mask = get_feature_mask()
    raw_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')]
    groups = {}
    
    for f in raw_files:
        if any(x in f for x in ["background", "negative"]): continue
        ore_type = f.split("_")[0]
        img = cv2.imread(os.path.join(TEMPLATE_DIR, f), 0)
        if img is not None:
            # TRANSFORM: Focus on sharp features only
            feature_img = process_for_features(cv2.resize(img, (48, 48)))
            if ore_type not in groups: groups[ore_type] = []
            groups[ore_type].append({'name': f, 'img': feature_img})

    ore_types = sorted(groups.keys())
    print(f"--- Feature-Emphasized Template Audit (v5.2) ---")
    print(f"[!] Using Edge Detection to isolate gem structures")

    # 1. CROSS-CLASS DISTINCTNESS
    print("\n[1] Checking Cross-Class Distinctness (Edges Only)...")
    for i, type_a in enumerate(ore_types):
        for j, type_b in enumerate(ore_types):
            if i >= j: continue
            
            max_collision = 0
            for t_a in groups[type_a]:
                for t_b in groups[type_b]:
                    res = cv2.matchTemplate(t_a['img'], t_b['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    max_collision = max(max_collision, res.max())
            
            # If they still match at > 0.80 on EDGES ONLY, they are truly identical
            if max_collision > 0.80:
                status = "!! STRUCTURAL COLLISION !!" if max_collision > 0.88 else "SIMILAR"
                print(f"  > {type_a} vs {type_b}: {max_collision:.3f} [{status}]")

    # 2. INTRA-CLASS COHESION
    print("\n[2] Checking Internal Structural Consistency...")
    for ore_type in ore_types:
        t_list = groups[ore_type]
        if len(t_list) < 2: continue
        scores = [cv2.matchTemplate(t_list[i]['img'], t_list[j]['img'], cv2.TM_CCORR_NORMED, mask=mask).max() 
                  for i in range(len(t_list)) for j in range(i+1, len(t_list))]
        print(f"  > {ore_type:5}: Edge Cohesion {np.mean(scores):.3f}")

if __name__ == "__main__":
    analyze_template_integrity_v52()