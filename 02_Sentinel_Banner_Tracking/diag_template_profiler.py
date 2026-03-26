# diag_template_profiler.py
# Purpose: Profile the entire template library to establish complexity bandwidths.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def apply_gamma_lift(img, gamma=0.6):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def run_library_profile():
    t_path = cfg.TEMPLATE_DIR
    results = []
    print(f"--- PROFILING TEMPLATE LIBRARY ---")
    
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        
        # Calculate raw complexity and lifted complexity
        raw_comp = cv2.Laplacian(img, cv2.CV_64F).var()
        lifted_comp = cv2.Laplacian(apply_gamma_lift(img), cv2.CV_64F).var()
        
        tier = f.split("_")[0]
        results.append({
            'filename': f,
            'tier': tier,
            'family': ''.join([i for i in tier if not i.isdigit()]),
            'raw_complexity': round(raw_comp, 2),
            'lifted_complexity': round(lifted_comp, 2)
        })

    df = pd.DataFrame(results)
    
    print("\n--- COMPLEXITY BANDWIDTH BY FAMILY ---")
    stats = df.groupby('family')['lifted_complexity'].agg(['min', 'max', 'mean', 'count'])
    print(stats)
    
    out_path = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit", "template_complexity_map.csv")
    df.to_csv(out_path, index=False)
    print(f"\nLibrary map saved to: {out_path}")

if __name__ == "__main__":
    run_library_profile()