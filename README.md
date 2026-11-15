# WasherVision — experimental OD + ID verification

Uses your trained YOLO detection weights (`models/best.pt`) and the existing webcam + planar homography workflow. **No motors are controlled.**

## Anaconda on Windows

```bat
conda env create -f environment.yml
conda activate washer_vision
python washer_vision.py
```

If the environment already exists, use `conda activate washer_vision` and `python washer_vision.py`.

Controls: `C` calibrate, click four corners in TL/TR/BR/BL order, `Enter` confirm, `R` reset, `S` save CSV, `Q` quit.

**Important:** Calibration dimensions must match the *actual* measured rectangle, not simply the sheet size. Default is 200 x 150 mm; override using `--width-mm` and `--height-mm`. Recalibrate after camera resolution, zoom, focus or mount changes. The camera's calibration is for a flat work plane only.

## What changed

- New `dimension_check.py` estimates outer and inner edges using contours and fitted ellipses. It assumes washers are darker than their background.
- The results panel shows estimated OD and ID, size consistency status, and optional alternative class.
- Measurements are deliberately rejected when the outer edge is irregular, no valid ellipse is found, or a washer overlaps another detection.
- If the inner hole is not reliably found, it shows `ID: ?` rather than inventing a measurement.
- N1 and W3 share OD 20 mm / ID 10 mm; dimensions alone cannot distinguish them.
- Original YOLO class predictions are **never automatically overwritten**. Suggested classes are experimental.
- CSV output now defaults to `washer_coordinates_od_id.csv` to avoid conflicting with older CSV columns.

## Known limitations and validation

- The inner opening can be misidentified because of shadows, reflections, printed grid lines or camera tilt.
- Circle/ellipse geometry is not guaranteed to be accurate for stacked, touching or partially occluded washers.
- The bounding-box centroid is still the reported X-Y point; the fitted ellipse is for size checking only.
- Check predicted OD/ID against actual washers (N1 20/10, W1 16/8, W2 18/11, W3 20/10, W4 22/7 mm) over several positions. Do not trust dimension corrections until errors are measured.
- `UNVERIFIED_FLAT` is not confirmation of a safe pickup. All motor movements require separate workspace-to-motor calibration and safety checks.

## Source acknowledgement

This is a local extension of the user's existing WasherVision project, which runs a trained Ultralytics YOLO detection model. It does not copy EdjeElectronics' scripts.

## Contour diagnostics (this version)

Run `python washer_vision.py --debug-edges` to show the color legend.
**Cyan** contours mark the measured outer edge, **magenta** contours mark the
inner opening, and the **red dot** is the YOLO bounding-box centroid.
The new `OUTER_EDGE_EXCEEDS_BOX` status rejects outer contours that exceed
the detected bounding box by over 20% in either direction. This is a
conservative diagnostic, **not** a guaranteed fix for W1; YOLO boxes
themselves can be too large or too small.

To test W1, place it alone on the grid, run the program, and inspect whether
the cyan line follows the actual outer rim and the magenta line follows the
hole. Repeat at different positions. If the cyan line follows a shadow,
improve lighting and contrast; if the YOLO box is too large, investigate
training images and labeling. Compare dimensions against a ruler/caliper.

Existing calibration is preserved if the camera geometry and resolution
are unchanged. Never use these size recommendations to actuate motors.

## Specification correction
W1 has **16 mm outer diameter** and **8 mm inner diameter** (previously listed incorrectly as 11 mm OD). This update changes the specification lookup and size-verification recommendations; the YOLO weights remain unchanged. Re-test the size checks with your physical washers.

## Combined classification and 30-frame voting (new)

Run on Windows or macOS (the camera backend is selected automatically):

```bash
conda activate washer_vision
python washer_vision.py --width-mm 127 --height-mm 101.6 --debug-edges
```

The `127 × 101.6 mm` dimensions are correct **only when the four calibration clicks span exactly 20 by 16 quarter-inch grid squares**. Press `C`, select TL/TR/BR/BL grid intersections and press Enter. Existing calibration files are not rescaled automatically.

The display now shows YOLO class, experimental size suggestion, and a 30-frame voted advisory class. A class is displayed as voted only after 30 observations, with at least 20 non-abstaining frames and at least 80% agreement over all 30 frames. Adjust with `--vote-window`, `--vote-min`, and `--vote-agreement`. Overlap, unreliable/missing OD or ID, or ambiguous dimensions cause abstention. N1 and W3 share dimensions: only the visual YOLO label is retained when confidence is at least 0.75, flagged `VISUAL_N1_W3_UNVERIFIED` (not a physically verified material classification).

Tracking uses nearest image centroid (up to 45 pixels between frames), independent of class, so class flips do not automatically reset voting. IDs are still temporary: if objects move, cross, or disappear, votes may be attributed to the wrong object. Keep the washers stationary during testing. The 30-frame vote is **not** a statistically calibrated confidence probability. For safety, **no motor commands are sent**.

The CSV filename is now `washer_coordinates_fusion.csv`, with additional fusion and vote fields. Preserve earlier CSVs for comparisons. Test the five known classes at several positions, and validate camera-to-workspace and workspace-to-motor calibration separately before any actuation.

## Experimental W2 tuning
For valid nonoverlapping OD/ID measurements, a W2 candidate can receive a
**provisional** suggestion if OD is 17.0–19.2 mm, ID is 9.5–11.8 mm,
W2 dimensional compatibility is at least 0.25, and exceeds every competing
class score by at least 10%. The 30-frame vote is unchanged. This rule was
chosen to investigate the observed W2 18.8/10.2 mm case and has NOT been
validated on independent arrangements. It may create false W2 suggestions.
Do not use advisory labels to control the gripper.
