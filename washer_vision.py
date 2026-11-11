"""Washer segmentation, planar calibration, and XY coordinate export.

Keys:
  c: select four calibration corners (TL, TR, BR, BL) on a frozen frame
  r: reset calibration points while selecting
  s: save currently detected target coordinates to CSV
  q: quit

The calibration target must lie on the washer working plane. Coordinates are
relative to its top-left corner, with +X right and +Y down in the target frame.
No motor commands are sent by this program.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"  # Change to yolov8n-seg.pt if appropriate
CALIBRATION_PATH = ROOT / "calibration.json"
CSV_PATH = ROOT / "washer_coordinates.csv"
CAMERA_INDEX = 0
CONFIDENCE = 0.50
TARGET_WIDTH_MM = 200.0   # Replace with measured calibration target width
TARGET_HEIGHT_MM = 150.0  # Replace with measured calibration target height
WINDOW = "Washer Vision | c calibrate | s save | q quit"


def save_calibration(H, image_size):
    payload = {
        "homography": H.tolist(),
        "image_size": list(image_size),
        "target_width_mm": TARGET_WIDTH_MM,
        "target_height_mm": TARGET_HEIGHT_MM,
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))


def load_calibration(image_size):
    if not CALIBRATION_PATH.exists():
        return None
    try:
        payload = json.loads(CALIBRATION_PATH.read_text())
        if payload["image_size"] != list(image_size):
            print("Camera resolution changed; recalibration required.")
            return None
        if (payload["target_width_mm"] != TARGET_WIDTH_MM or
                payload["target_height_mm"] != TARGET_HEIGHT_MM):
            print("Target dimensions changed; recalibration required.")
            return None
        H = np.asarray(payload["homography"], dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            return None
        print("Loaded calibration from", CALIBRATION_PATH)
        return H
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Could not load calibration:", exc)
        return None


def calibrate(frame):
    """User selects four target corners in TL, TR, BR, BL order."""
    points = []
    window = "Calibration - click TL, TR, BR, BL | r reset | Esc cancel"

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 5, (0, 165, 255), -1)
            cv2.putText(canvas, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        prompt = ("Click: " + ["top-left", "top-right", "bottom-right", "bottom-left"][len(points)]
                  if len(points) < 4 else "Press ENTER to confirm | r to retry")
        cv2.putText(canvas, prompt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            points.clear()
        elif key == 27:
            cv2.destroyWindow(window)
            return None
        elif key in (10, 13) and len(points) == 4:
            break
    cv2.destroyWindow(window)
    src = np.asarray(points, dtype=np.float32)
    dst = np.asarray([[0, 0], [TARGET_WIDTH_MM, 0],
                      [TARGET_WIDTH_MM, TARGET_HEIGHT_MM],
                      [0, TARGET_HEIGHT_MM]], dtype=np.float32)
    if abs(cv2.contourArea(src)) < 100:
        print("Calibration corners are too close together.")
        return None
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(H).all():
        return None
    return H


def find_washers(frame, model, H):
    result = model.predict(source=frame, conf=CONFIDENCE, verbose=False)[0]
    detections = []
    if result.masks is None:
        return detections
    # Polygon points from masks.xy are expressed in original image pixels.
    for i, polygon in enumerate(result.masks.xy):
        if len(polygon) < 3:
            continue
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.rint(polygon).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        moments = cv2.moments(mask, binaryImage=True)
        if moments['m00'] <= 0:
            continue
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        x_mm = y_mm = None
        if H is not None:
            p = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(p, H)[0, 0]
            if np.isfinite(transformed).all():
                x_mm, y_mm = map(float, transformed)
        confidence = (float(result.boxes.conf[i]) if result.boxes is not None
                      and i < len(result.boxes.conf) else None)
        detections.append((cx, cy, x_mm, y_mm, confidence))
        cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
        cv2.circle(frame, (round(cx), round(cy)), 5, (0, 0, 255), -1)
        label = (f"W{i+1}: ({x_mm:.1f}, {y_mm:.1f}) mm" if x_mm is not None
                 else f"W{i+1}: ({cx:.0f}, {cy:.0f}) px")
        cv2.putText(frame, label, (max(0, round(cx) - 50), max(20, round(cy) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return detections


def append_csv(detections):
    valid = [d for d in detections if d[2] is not None]
    if not valid:
        print("No calibrated washer positions to save.")
        return
    new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open('a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['timestamp', 'washer_id', 'pixel_x', 'pixel_y',
                             'workspace_x_mm', 'workspace_y_mm', 'confidence'])
        timestamp = datetime.now().isoformat(timespec='seconds')
        for i, (cx, cy, x, y, conf) in enumerate(valid, start=1):
            writer.writerow([timestamp, i, round(cx, 2), round(cy, 2),
                             round(x, 2), round(y, 2),
                             '' if conf is None else round(conf, 3)])
    print(f"Saved {len(valid)} washer(s) to {CSV_PATH}")


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened. Try CAMERA_INDEX = 1.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    H = None
    frame_size = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame unavailable.")
                break
            size = (frame.shape[1], frame.shape[0])
            if size != frame_size:
                frame_size = size
                H = load_calibration(size)
            annotated = frame.copy()
            detections = find_washers(annotated, model, H)
            status = ("CALIBRATED - workspace coordinates (mm)" if H is not None
                      else "NOT CALIBRATED - press c")
            cv2.putText(annotated, status, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(WINDOW, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_H = calibrate(frame)
                if new_H is not None:
                    H = new_H
                    save_calibration(H, size)
                    print("Calibration saved. Verify using known points.")
            elif key == ord('s'):
                append_csv(detections)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
