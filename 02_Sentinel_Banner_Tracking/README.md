# Ore Floor Tracker: Step 1 (Sprite Homing)

This module handles the spatial detection of mining events. It is designed to identify exactly when and where a player is mining in Row 1 (Slots 0–5) and Row 2 (Slot 11) to facilitate floor-start identification.

## 1. The Physics: The Consensus Grid

The system operates on a "Consensus Grid" derived from dual-coordinate calibration. It strictly decouples the **AI Match Point** (the top-left anchor used for calculation) from the **HUD Visual Center** (the torso/shadow point used for verification).

| Constant | Value | Description |
| :--- | :--- | :--- |
| **Origin (S0 AI)** | `(11, 225)` | The Top-Left pixel where the AI begins its Slot 0 match. |
| **Grid Step** | `59.0px` | The precise horizontal and vertical distance between tile centers. |
| **Stand Offset** | `41.0px` | Lateral distance from the Ore Center to the Player Stand position. |
| **Stand Flip** | `82.0px` | The total shift required when the player faces Left (Slot 11) vs. Right (Slots 0-5). |

### Coordinate Translation

To verify alignment visually, use the following translation:
* $X_{HUD} = X_{AI} + 20$
* $Y_{HUD} = Y_{AI} + 30$

---

## 2. Core Logic: The Hybrid Staircase (v2.8)

Standard template matching fails in high-activity zones due to "Dig Stage" banners and scrolling UI text. `sprite_sequencer.py` employs two specialized mechanisms to ensure 100% coverage of the "Happy Path":

### A. Hybrid Redundancy Sensor
For every slot, the AI runs two simultaneous checks and takes the **Maximum** confidence:
1. **Full-Body Match**: Uses the entire $40 \times 60$ sprite. Offers high entropy but is vulnerable to UI text overlapping the player's head.
2. **Bottom-Half Match**: Targets only the feet and ground shadow (bottom $30\text{px}$). Offers lower entropy but is immune to top-row UI interference.

### B. The Staircase Thresholds
Detection confidence naturally decays from left to right as background noise increases. We use a "Staircase" model to prevent false negatives on the right edge while maintaining strictness on the left:
* **Slot 0**: `0.90` (Pristine)
* **Slot 3**: `0.78` (Moderate Noise)
* **Slot 5**: `0.72` (Maximum Sensitivity)
* **Slot 11**: `0.82` (Clean Row 2)

---

## 3. Re-Calibration Guide (New Datasets)

If applying these scripts to a new capture buffer with different resolutions or UI scaling, follow this diagnostic pipeline:

### Step A: Verify Coordinates
**Script:** `sprite_consensus_verification.py`  
Run this first. If the yellow HUD boxes do not perfectly "wrap" the player sprite in the output images, your `AI_S0_X/Y` origin or `STEP` constants require adjustment.

### Step B: Audit Signal-to-Noise (SNR)
**Script:** `sprite_right_side_pulse_audit.py`  
If Slots 4 and 5 report zero hits, run this to visualize the "Pulses." It compares Full-Body vs. Bottom-Half scores to identify if UI text is suppressing the signal.

### Step C: Tune Thresholds
**Script:** `sprite_hybrid_sensitivity_audit.py`  
Generates a mean confidence report for every slot. Use the `Bottom_Half_Avg` column to set the new floor for your Staircase Thresholds.

### Step D: Negative Space Accounting
**Script:** `sprite_negative_audit.py`  
The final check. This analyzes every frame the sequencer *ignored*. It categorizes discards into Rows 1–4 to ensure no high-confidence mining events were left behind.

---

## 4. File Manifest

| File | Role |
| :--- | :--- |
| **`sprite_sequencer.py`** | **Production.** Generates the `sprite_homing_run_X.csv`. |
| **`sprite_negative_audit.py`** | **Audit.** Validates exclusion logic and accounts for all buffer frames. |
| **`sprite_staircase_threshold_proof.py`** | **Diagnostic.** Validates the per-slot threshold model. |
| **`sprite_ultimate_audit.py`** | **Diagnostic.** Verifies Slot 11 Stand-Flip and row transitions. |
| **`project_config.py`** | **Global.** Stores paths, directory structures, and environment settings. |