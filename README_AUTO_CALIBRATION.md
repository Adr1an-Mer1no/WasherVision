# Automatic OD/ID measurement calibration (experimental)

This version **does not know the washer class automatically during calibration**. You must supply the true class for each isolated washer. It learns a single OD correction and a single ID correction shared across all classes; it never applies a class-specific W2 fudge factor. YOLO weights are unchanged. Motor commands remain disabled.

## Collect known washer samples

Keep the camera fixed and use the same graph-paper calibration as before. Open Terminal inside this repository:

```bash
conda activate washer_vision
python washer_vision.py --width-mm 127 --height-mm 101.6 --debug-edges --known-class W1
```

Place **only W1** on the graph paper, wait for at least 30 valid measurement frames, press **K** to save a median OD/ID sample. Move the washer to a different position, wait 30 valid frames, and press K again. Repeat for **three different positions**. Restart with `--known-class W2`, then W3, W4, N1, collecting at least **three independent positions per class** (15 samples total). Do not relabel YOLO predictions as ground truth: use the physical washer you actually placed. Samples append to `dimension_samples.csv`. Delete that file to restart the collection.

**Important:** K records the preceding 30 valid frames and clears the buffer. Do not press K immediately after moving a washer; wait until the old frames have been replaced. Keep each washer flat and isolated, and do not move the camera. Recalibrate if camera position changes.

## Fit and validate

```bash
python dimension_calibration.py --samples dimension_samples.csv --output dimension_calibration.json
```

The fitter reserves the last sample of each class for a holdout test, fits a shared regularized linear correction to the other samples, and refuses to save a model if the held-out average absolute error is worse. This is a small engineering calibration, not a proof of classifier accuracy. Collect more samples and test on additional placements before using it in a robot.

## Run with correction

```bash
python washer_vision.py --width-mm 127 --height-mm 101.6 --debug-edges
```

If `dimension_calibration.json` exists, the program automatically loads it and displays raw and corrected dimensions. Classification uses corrected dimensions. To compare with uncorrected behavior, temporarily rename `dimension_calibration.json` and restart. **Do not use motor control from these recommendations.**

A corrected size measurement cannot guarantee W2 classification, especially because W2 and W3 are close in size and YOLO can be wrong. N1 and W3 have identical nominal OD/ID and require visual evidence.
