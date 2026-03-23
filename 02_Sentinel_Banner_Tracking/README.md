# Ore Data Collection Pipeline (v2.1)

Linear 5-step pipeline to convert raw video captures into validated 110-floor datasets.

## 1. Technical Constants (Asc1 Baseline)

| Constant | Value | Description |
| :--- | :--- | :--- |
| **Grid Origin** | `(74, 261)` | Validated center of Slot 0. |
| **Grid Step** | `59.0px` | Distance between ores. |
| **Complexity Gate**| `500` | Min Laplacian variance to distinguish Shadow from Active. |
| **Signal Floor** | `0.30` | Min confidence required for an Ore vote to count. |

---

## 2. Pipeline Execution Order

### Step 0: Grid Calibration
**Script:** `step0_calibrate_grid.py`
Run for every new dataset to find physical resolution and alignment.
* **Tip:** If the script suggests **Y=320**, it found Row 2. Subtract 59 to reach the true **Y=261** baseline.

### Step 1: Sprite Homing
**Script:** `step1_sprite_homing.py`
* **Output:** `sprite_homing_run_X.csv`.

#### 🔍 Step 1 Quality Control (Optional)
**Script:** `step1_audit_negatives.py`
Run this if you suspect the Homing script is missing floors. 
* **Action:** It scans every frame Step 1 *ignored* and looks for high-confidence player matches.
* **Interpretation:** If the script finds many "False Negatives" (confidence > 0.75) in Row 1, you should lower the `STAIRCASE` thresholds in `step1_sprite_homing.py`.

### Step 2: Frame-Level DNA
**Script:** `step2_frame_dna.py`
Generates raw 24-bit signatures (Occupied vs Empty) for every frame.

### Step 3: Floor Segmentation
**Script:** `step3_floor_boundaries.py`
Defines floor start/end timings based on Player movement and DNA stability.

### Step 4: Identification Phase
* **Step 4.1: Floor Occupancy (`step4_1_floor_occupancy.py`)**: Creates the initial 1/0 mask.
* **Step 4.2: Tier Consensus (`step4_2_tier_consensus.py`)**: Forensic voting engine (v6.4).

### Step 5: Integrity Audit
**Script:** `step5_integrity_audit.py`
Mathematical QC against biome restrictions and forensic call spots.

---

## 3. Dataset Migration (Repeatability)
To process a new capture buffer (e.g., moving from Run 1 to Run 2):

1. **Update Project Config:** Open `project_config.py` and change the default value in `get_buffer_path(buffer_id=2)`.
2. **Step 0 Verify:** Run `step0_calibrate_grid.py`. Check the image in `Data_04_Calibration_Vault`. If the grid shifted (due to different window positions), update the `ORE0` constants in config.
3. **Execute 1–5:** Run the scripts in sequence. Step 1 will automatically name its output `sprite_homing_run_2.csv` based on your config change.