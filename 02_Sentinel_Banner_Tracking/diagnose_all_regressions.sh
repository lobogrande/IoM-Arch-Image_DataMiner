#!/bin/bash
# Diagnose all baseline regressions using the working diagnose_specific_slot.py

echo "========================================================================"
echo "RARE1 → COM2 REGRESSIONS (sample: Floor 18 R3_S1)"
echo "========================================================================"
python diagnose_specific_slot.py 18 3 1 --max-frames 15

echo ""
echo "========================================================================"
echo "EPIC1 → DIRT1 REGRESSIONS (sample: Floor 7 R2_S0)"
echo "========================================================================"
python diagnose_specific_slot.py 7 2 0 --max-frames 15

echo ""
echo "========================================================================"
echo "RARE1 → LOW_CONF REGRESSIONS (sample: Floor 21 R1_S2)"
echo "========================================================================"
python diagnose_specific_slot.py 21 1 2 --max-frames 15

echo ""
echo "========================================================================"
echo "RARE1 → COM1 REGRESSIONS (sample: Floor 5 R2_S4)"
echo "========================================================================"
python diagnose_specific_slot.py 5 2 4 --max-frames 15
