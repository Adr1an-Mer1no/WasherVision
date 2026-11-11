# WasherVision — YOLO11 washer centroid & workspace XY

An Anaconda-ready computer-vision prototype for a **soft robotic gripper**. Uses the user's trained YOLO **object-detection** weights (not segmentation) to find bounding-box centers, applies a planar homography for an angled camera, and logs measured workspace XY positions. **It never moves motors.**

## Included
- `washer_vision.py`: live webcam or static-image YOLO inference, mouse calibration, display, CSV export
- `vision_utils.py`: centroid, homography, overlap warning, washer class lookup
- `models/best.pt`: trained weights supplied with the project (extracted from uploaded `my_model.zip`)
- `environment.yml`: Anaconda dependencies
- `tests/test_geometry.py`: geometry sanity checks

**No full image dataset included:** the uploaded dataset is ~190 MB. Keep your original `data.zip.zip` separately if retraining.

## Washer classes
| Class | Material | OD (mm) | ID (mm) |
|---|---|---:|---:|
| N1 | Nylon | 20 | 10 |
| W1 | Metal | 11 | 8 |
| W2 | Metal | 18 | 11 |
| W3 | Metal | 20 | 10 |
| W4 | Metal | 22 | 7 |

N1 and W3 have the same dimensions but different materials. Class predictions come from the model, not from geometric measurements.

## Install (Windows Anaconda Prompt)
```bash
cd path\to\WasherVision_Complete
conda env create -f environment.yml
conda activate washer_vision
python washer_vision.py --camera 0 --confidence 0.50
```
If `conda env create` reports an existing environment, use `conda env update -f environment.yml --prune` instead.

Optional image test: `python washer_vision.py --image path\to\image.jpg`.

## Calibration: essential for angled camera
1. Secure camera at its final fixed mounting position (approximately 13 inches high). Fix focus, zoom, resolution and angle; recalibrate after changing any of these.
2. Put a **measured rectangular target** flat on the **same plane as the washer bottoms**. Its real-world X and Y edges must define the coordinate axes you want.
3. Run script with measured target dimensions, e.g. `--width-mm 200 --height-mm 150` (examples only; use actual values).
4. Press **C**. Click four corners **TL, TR, BR, BL relative to the target's physical coordinate orientation**, not arbitrary image screen corners. Press **Enter** to save `calibration.json`.
5. Verify against several independently measured points across the workspace. The resulting (0,0) is the target's first corner; positive X runs toward second corner and positive Y toward fourth corner.

**Keys:** C = start calibration; left-click = add corner; R = clear clicked corners; Enter = confirm; S = append detected coordinates to CSV; Q/Esc = quit.

## Stacked / overlapping washers
YOLO detection boxes can still report a centroid for a **partially visible** washer, but it may **not be the true washer center**. The script marks overlapping boxes as `REVIEW_OVERLAP` and every other detection `UNVERIFIED_FLAT`. These are screening flags, **not stack detection**. A fully occluded washer may be missed entirely. **Do not use these coordinates for unattended pickup.** In future, consider instance segmentation, depth sensing, height measurement, and a grasp-safety gate to handle stacked objects.

## Motors: not yet connected
The XY output is in the **calibration target's workspace frame**, not automatically in motor coordinates. Before motion, establish motor homing, frame alignment, tool-center offsets, axis travel limits, a safe Z approach, collision/stack checks, and an emergency stop. The CSV includes `motor_command_sent=False` to make the prototype status explicit.

## Accuracy limits
- Homography is valid for **one planar surface**. Stacked washers are above that plane and their mapped XY can be wrong due to parallax.
- Detection-box center may differ from true washer center due to occlusion or perspective.
- Four-point homography does not correct lens distortion. For precision, perform intrinsic camera calibration/undistortion and repeat homography on undistorted frames.
- Confidence is not geometric accuracy. Validate measured XY error in millimeters at multiple points.

## Tests
```bash
python -m pytest -q
```

## Publish to GitHub
Create an empty repository named `WasherVision`, then from the extracted folder:
```bash
git init
git add .
git commit -m "Initial washer vision prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/WasherVision.git
git push -u origin main
```
The included `models/best.pt` is about 19 MB. GitHub can accept it as a regular file, though Git LFS is a better long-term option for model binaries. Only publish data and model weights you have permission to share.

## Reference inspiration
- EdjeElectronics: https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models
- YOLO training tutorial: https://www.youtube.com/watch?v=r0RspiLG260

This is an independently written prototype based on the project's workflow; it does not copy the referenced scripts.
