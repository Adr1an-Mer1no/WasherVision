# WasherVision — YOLO Centroid and XY Calibration

Python project for an overhead webcam (slightly angled) to detect washers using a **custom-trained YOLO segmentation model**, compute mask centroids, map image points to a flat workspace using a four-point homography, and export XY positions in millimeters. **No motor commands are issued.**

## Requirements
- Anaconda or Miniconda
- Webcam
- Custom segmentation weights named `best.pt` (not included)
- A measured rectangular calibration target lying on the same plane as the washers

## Install (Anaconda Prompt)
```bash
conda env create -f environment.yml
conda activate washer_vision
```

## Configure
1. Copy your trained YOLO segmentation weights into this directory as `best.pt`.
2. Open `washer_vision.py`. Set `TARGET_WIDTH_MM` and `TARGET_HEIGHT_MM` to the **actual measured** rectangle dimensions (defaults 200 × 150 mm are only examples).
3. Set `CAMERA_INDEX` to 0 or 1 as needed.
4. Keep the camera fixed and use the same capture resolution for calibration and detection.

## Run
```bash
python washer_vision.py
```

## Controls
- **C**: freeze frame and select four corners of calibration rectangle (top-left, top-right, bottom-right, bottom-left, relative to desired target axes)
- **R**: reset selected calibration corners
- **Enter**: confirm four corners
- **Esc**: cancel calibration
- **S**: append calibrated detections to `washer_coordinates.csv`
- **Q**: quit

Calibration saves `calibration.json` locally; these generated files are excluded from Git.

## Coordinate conventions
- The calibration rectangle's top-left corner is `(0,0)` mm.
- +X runs toward its top-right corner; +Y runs toward its bottom-left corner.
- A homography corrects **planar perspective** from a slightly angled camera, but **not lens distortion**.
- Coordinates are **workspace positions**, not yet gripper/motor coordinates. A separate camera-to-motor frame calibration is needed before commanding motion.

## Safety and limitations
- Do not use XY values to drive motors until you have measured the mapping to motor coordinates and validated position accuracy at multiple points.
- Move at safe Z clearance during XY testing, use homing, soft limits and emergency stop.
- All washers should lie on the same plane as the calibration target.
- `best.pt` must be a **segmentation** model trained for your washers; generic `yolov8n-seg.pt` weights may not detect washers reliably.
- The script uses filled segmentation polygons for centroids; for annular washer masks where preserving the hole is important, adapt centroid calculation to native masks.
