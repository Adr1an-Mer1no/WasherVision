# W2 classifier fix and diagnostic

The user-uploaded `class_fusion.py` was missing the `PROVISIONAL_W2_SIZE_ADVANTAGE` rule. This package restores that rule and displays W2 and W3 relative dimensional scores in the side panel. The values are **not** probabilities.

Run from this folder:

```bash
conda activate washer_vision
python washer_vision.py --width-mm 127 --height-mm 101.6 --debug-edges
```

If calibration was made with the same 20 x 16 graph-paper squares, keep the camera fixed. Confirm 30 stable frames. Never use advisory classifications for automatic motor actuation without separate validation.
