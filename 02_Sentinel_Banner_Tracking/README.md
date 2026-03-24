# Ore Data Collection Pipeline (v3.0)

Linear 8-step pipeline to convert raw video captures into validated 110-floor datasets.

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
Identifies exactly when and where the player is mining in the grid.
* **Output:** `sprite_homing_run_X.csv` (The Player Source of Truth).

#### 🔍 Step 1 Quality Control (Optional)
**Script:** `step1_audit_negatives.py`
Run this if you suspect the Homing script is missing floors. 
* **Action:** It scans every frame Step 1 *ignored* and looks for high-confidence player matches.
* **Interpretation:** If the script finds many "False Negatives" (confidence > 0.70) in Row 1, you should lower the `STAIRCASE` thresholds in `step1_sprite_homing.py`.

### Step 2: Frame-Level DNA
**Script:** `step2_frame_dna.py`
* **Action:** Generates raw 24-bit signatures (Occupied vs Empty) for every frame using background correlation.
* **Output:** `dna_sensor_run_X.csv`.

### Step 3: Floor Segmentation
**Script:** `step3_floor_segmentation.py`
* **Action:** Groups frames into distinct floor blocks based on player slot-reversals and Row 4 DNA stability. Identifies the "Anchor" frame of the floor.
* **Output:** `floor_start_candidates_run_X.csv`.

### Step 4: Boundary Verification
**Script:** `step4_boundary_verifier.py`
* **Action:** Scans backward from Step 3 anchors to find the exact frame the DNA shifted (the true floor transition). Also injects Frame 0 if the run started late.
* **Output:** `final_floor_boundaries_run_X.csv` and visual verification proofs.

### Step 5: Floor Occupancy Masking
**Script:** `step5_floor_occupancy.py`
* **Action:** Scans a 150-frame window after floor arrival to create the finalized 1/0 map indicating which slots have ores. Boss floors automatically bypass this and load directly from project config.
* **Output:** `floor_dna_inventory_run_X.csv`.

### Step 6: Tier Consensus
**Script:** `step6_tier_consensus.py`
* **Action:** The forensic identification engine. Uses the Step 5 mask, Step 1 Homing data (to exclude player-obstructed frames), and Temporal Consensus Voting to achieve 100% identification accuracy.
* **Output:** `floor_ore_inventory_run_X.csv` and visual proofs.

### Step 7: Integrity Audit
**Script:** `step7_integrity_audit.py`
* **Action:** Mathematical Quality Control. Validates final IDs against biome restrictions, limits forensic calls to user-approved spots, and outputs the Tier Distribution Summary.

---

## 3. Dataset Migration (Repeatability)
To process a new capture buffer (e.g., moving from Run 1 to Run 2):

1. **Update Project Config:** Open `project_config.py` and change the default value in `get_buffer_path(buffer_id=2)`.
2. **Step 0 Verify:** Run `step0_calibrate_grid.py`. Check the image in `Data_04_Calibration_Vault`. If the grid shifted (due to different window positions), update the `ORE0` constants in config.
3. **Execute 1–7:** Run the scripts in sequence. Because paths are dynamic, the scripts will automatically name their outputs (e.g., `sprite_homing_run_2.csv`) based on your config change.