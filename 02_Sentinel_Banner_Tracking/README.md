# Ore Data Collection Pipeline

This project converts raw video captures of mining gameplay into a structured 110-floor dataset of ore distributions. It uses a 4-step computer vision pipeline optimized for high-noise environments (Lava Biome).

## Pipeline Steps

### Step 0: Grid Calibration
**Script:** `step0_calibrate_grid.py`
Before running a new dataset, the AI must discover the physical resolution and grid alignment.
* **Input:** `capture_buffer_X/frame_0.png`
* **Action:** Brute-force sweeps scale (0.7–1.5) to find the "Geometric Truth."
* **Output:** Validated `ORE0_X`, `ORE0_Y`, and `STEP` constants. Update these in `project_config.py` or the subsequent scripts.

### Step 1: Sprite Homing
**Script:** `step1_sprite_homing.py`
Identifies exactly when and where the player is mining in the grid.
* **Logic:** Uses the **Hybrid Staircase Sensor** (Max of Full-Body vs. Bottom-Half) to detect the player through UI text.
* **Output:** `sprite_homing_run_X.csv`.

### Step 2: DNA & Occupancy
**Script:** `step2_dna_occupancy.py`
Determines which of the 24 slots contain an object (Ore or Player) vs. empty ground.
* **Logic:** Uses a "Valley Threshold" (0.75) against background templates to generate a 24-bit signature per frame.
* **Output:** `dna_sensor_final.csv`.

### Step 3: Floor Segmentation
**Script:** `step3_floor_segmentation.py`
Groups frames into distinct floors using Kinematic laws.
* **Laws:** 
    1. **Slot Reversal:** If player moves from right to left, the floor reset.
    2. **DNA Immutability:** If Row 4 changes significantly, a new floor has arrived.
* **Output:** `final_floor_boundaries.csv`.

### Step 4: Identification Phase

#### Step 4.1: Floor Occupancy
**Script:** `step4_1_floor_occupancy.py`
Determines which slots are occupied at the start of each floor.
* **Input:** `final_floor_boundaries.csv` (Step 3)
* **Action:** Scans a 150-frame window after floor arrival. Uses pure background matching to differentiate between static ground and "something" (Ore/Player).
* **Output:** `floor_dna_inventory.csv` (A 24-slot 1/0 mask used as the filter for Step 4.2).

#### Step 4.2: Tier Consensus
**Script:** `step4_2_tier_consensus.py`
The final forensic identification engine.
* **Logic:** Applies the 1/0 mask from 4.1. Samples the first 40 frames and uses Temporal Consensus Voting + Homing Hard-Gates to identify tiers with 100% accuracy.

### Step 5: Integrity Audit
**Script:** `step5_integrity_audit.py`
A mathematical check to ensure no "impossible" ores exist.
* **Checks:** Validates results against `ORE_RESTRICTIONS` and verifies the 3 authorized forensic locations (F2, F3, F7).

---

## Technical Constants (Lava Biome Baseline)

| Constant | Value | Description |
| :--- | :--- | :--- |
| **Grid Origin** | `(74, 261)` | Validated center of Slot 0. |
| **Grid Step** | `59.0px` | Distance between ores. |
| **Complexity Gate**| `500` | Minimum Laplacian variance to distinguish Shadow from Active. |
| **Signal Floor** | `0.30` | Minimum confidence required for a vote to count. |

## Dataset Migration (Repeatability)
To process a new dataset (e.g., Run 1):
1. Update `BUFFER_ID = 1` in `project_config.py`.
2. Run **Step 0** to confirm the grid hasn't shifted more than 1-2 pixels.
3. Execute **Steps 1 through 5** in sequence.