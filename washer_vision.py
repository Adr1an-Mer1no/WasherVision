"""WasherVision: YOLO11 detection, planar XY calibration, display and stability checks.
No motor commands are sent. Requires vision_utils.py from the WasherVision repository.
"""
import argparse
import csv
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from vision_utils import (WASHERS, box_centroid, calibrate, pixel_to_workspace,
                          overlaps, save_calibration, load_calibration)

ROOT = Path(__file__).resolve().parent
COLORS = {'high': (70, 205, 90), 'review': (0, 190, 255), 'low': (80, 100, 255)}


def text(img, message, pos, scale=.55, color=(235, 235, 235), thickness=1):
    cv2.putText(img, str(message), pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def assign_tracks(detections, tracks, next_id, max_distance=45):
    """Associate current detections to previous-frame centroids, by class and proximity.
    IDs are temporary; reappearing or crossing objects may receive new IDs.
    """
    candidates = []
    for i, det in enumerate(detections):
        for tid, track in tracks.items():
            if det['name'] != track['name']:
                continue
            distance = np.hypot(det['cx']-track['cx'], det['cy']-track['cy'])
            if distance <= max_distance:
                candidates.append((distance, i, tid))
    used_dets, used_tracks = set(), set()
    for _, i, tid in sorted(candidates):
        if i not in used_dets and tid not in used_tracks:
            detections[i]['track_id'] = tid
            used_dets.add(i)
            used_tracks.add(tid)
    for i, det in enumerate(detections):
        if i not in used_dets:
            det['track_id'] = next_id
            next_id += 1
    return next_id


def main():
    parser = argparse.ArgumentParser(description='WasherVision improved live detection')
    parser.add_argument('--model', default=str(ROOT / 'models' / 'best.pt'))
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=.50)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--camera-width', type=int, default=1920)
    parser.add_argument('--camera-height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width-mm', type=float, default=200)
    parser.add_argument('--height-mm', type=float, default=150)
    parser.add_argument('--calibration', default=str(ROOT / 'calibration.json'))
    parser.add_argument('--csv', default=str(ROOT / 'washer_coordinates.csv'))
    parser.add_argument('--image', default=None)
    parser.add_argument('--history', type=int, default=30)
    args = parser.parse_args()

    if not Path(args.model).exists():
        parser.error(f'Model missing: {args.model}')
    if args.history < 2:
        parser.error('--history must be at least 2')
    model = YOLO(args.model)
    cap = None
    if args.image:
        source = cv2.imread(args.image)
        if source is None:
            parser.error(f'Cannot read image: {args.image}')
    else:
        # DirectShow is Windows-specific; use the default backend elsewhere.
        import os
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.camera, backend)
        if not cap.isOpened():
            parser.error(f'Cannot open camera {args.camera}')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        ok, source = cap.read()
        if not ok:
            cap.release()
            parser.error('Cannot read webcam')
        print('Camera actual resolution:', source.shape[1], 'x', source.shape[0])
        print('Camera reported FPS:', cap.get(cv2.CAP_PROP_FPS))

    size = (source.shape[1], source.shape[0])
    H = None
    calpath = Path(args.calibration)
    if calpath.exists():
        try:
            H = load_calibration(calpath, size)
            print('Loaded calibration:', calpath)
        except (ValueError, KeyError) as exc:
            print('Calibration not used:', exc)

    corners = []
    calibrating = False
    last_rows = []
    tracks = {}
    histories = defaultdict(lambda: deque(maxlen=args.history))
    next_id = 1
    frame_number = 0
    panel_width = 450
    window = 'WasherVision | C calibrate | ENTER confirm | R reset | S save | Q quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def click(event, x, y, flags, param):
        # Calibration clicks refer to the original frame, not the right-side panel.
        if calibrating and event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            if 0 <= x < size[0] and 0 <= y < size[1]:
                corners.append((x, y))

    cv2.setMouseCallback(window, click)
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print('Webcam frame unavailable')
                    break
            else:
                frame = source.copy()
            if (frame.shape[1], frame.shape[0]) != size:
                raise RuntimeError('Image resolution changed; restart and recalibrate')
            frame_number += 1
            display = frame.copy()
            panel = np.full((size[1], panel_width, 3), (32, 35, 40), dtype=np.uint8)
            last_rows = []
            text(panel, 'WASHERVISION', (20, 38), .85, (255, 255, 255), 2)
            text(panel, 'Calibrated' if H is not None else 'NOT CALIBRATED',
                 (20, 72), .62, (95, 220, 125) if H is not None else (0, 200, 255), 2)
            text(panel, 'ID  CLASS  CONF     X / Y (mm)', (20, 118), .52)
            text(panel, '------------------------------------', (20, 135), .48, (130, 140, 145))

            if calibrating:
                text(display, 'Click TL, TR, BR, BL of the physical rectangle',
                     (15, 36), .75, (0, 255, 255), 2)
                for i, (x, y) in enumerate(corners):
                    cv2.circle(display, (x, y), 7, (0, 140, 255), -1)
                    text(display, str(i + 1), (x + 10, y - 8), .7, (0, 140, 255), 2)
                text(panel, 'CALIBRATION MODE', (20, 180), .65, (0, 255, 255), 2)
                text(panel, f'Corners selected: {len(corners)}/4', (20, 218), .58)
            else:
                result = model.predict(frame, conf=args.confidence,
                                       imgsz=args.imgsz, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    clsids = result.boxes.cls.cpu().numpy().astype(int).tolist()
                    scores = result.boxes.conf.cpu().numpy().tolist()
                    for i, (box, clsid, score) in enumerate(zip(boxes, clsids, scores)):
                        cx, cy = box_centroid(box)
                        name = str(model.names[clsid])
                        overlap = overlaps(box, boxes, i)
                        x_mm = y_mm = None
                        if H is not None:
                            x_mm, y_mm = pixel_to_workspace(cx, cy, H)
                        detections.append(dict(box=box, name=name, score=float(score),
                                               cx=cx, cy=cy, x_mm=x_mm, y_mm=y_mm,
                                               overlap=overlap))

                next_id = assign_tracks(detections, tracks, next_id)
                updated_tracks = {}
                for i, det in enumerate(detections):
                    tid = det['track_id']
                    updated_tracks[tid] = dict(name=det['name'], cx=det['cx'], cy=det['cy'])
                    score = det['score']
                    confidence_level = 'high' if score >= .85 else 'review' if score >= .65 else 'low'
                    status = ('REVIEW_OVERLAP' if det['overlap'] else
                              'REVIEW_CONFIDENCE' if confidence_level != 'high' else
                              'UNVERIFIED_FLAT')
                    color = COLORS['review'] if det['overlap'] else COLORS[confidence_level]
                    x1, y1, x2, y2 = map(int, det['box'])
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(display, (round(det['cx']), round(det['cy'])), 5,
                               (0, 0, 255), -1)
                    label = f"#{tid} {det['name']} {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .62, 2)
                    label_x = min(max(0, x1), max(0, size[0] - tw - 8))
                    label_y = max(26, y1 - 9)
                    cv2.rectangle(display, (label_x - 2, label_y - th - 6),
                                  (label_x + tw + 5, label_y + 5), (30, 35, 38), -1)
                    text(display, label, (label_x, label_y), .62, color, 2)

                    if det['x_mm'] is not None:
                        histories[tid].append((det['x_mm'], det['y_mm']))
                    else:
                        histories[tid].clear()
                    history = histories[tid]
                    sx = sy = None
                    if len(history) >= 2:
                        sx, sy = np.std(np.array(history), axis=0, ddof=1)
                    if det['x_mm'] is None:
                        coords = 'UNCALIBRATED'
                    else:
                        coords = f"{det['x_mm']:.1f}, {det['y_mm']:.1f}"
                    row_y = 165 + i * 102
                    if row_y + 82 < size[1]:
                        text(panel, f"#{tid} {det['name']}  {score:.2f}   {coords}",
                             (20, row_y), .56, color, 2)
                        text(panel, status, (20, row_y + 28), .47, color)
                        if sx is not None:
                            text(panel, f'Stability ({len(history)}/{args.history}): '
                                 f'sX={sx:.2f}, sY={sy:.2f} mm',
                                 (20, row_y + 54), .45, (190, 205, 215))
                        else:
                            text(panel, f'Stability: {len(history)}/{args.history} frames',
                                 (20, row_y + 54), .45, (190, 205, 215))
                    spec = WASHERS.get(det['name'], {})
                    last_rows.append(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        class_name=det['name'], confidence=round(score, 4),
                        pixel_x=round(det['cx'], 2), pixel_y=round(det['cy'], 2),
                        workspace_x_mm=round(det['x_mm'], 2) if det['x_mm'] is not None else '',
                        workspace_y_mm=round(det['y_mm'], 2) if det['y_mm'] is not None else '',
                        material=spec.get('material', 'unknown'),
                        od_mm=spec.get('od_mm', ''), id_mm=spec.get('id_mm', ''),
                        status=status, motor_command_sent=False,
                        track_id=tid, stability_samples=len(history),
                        std_x_mm=round(float(sx), 3) if sx is not None else '',
                        std_y_mm=round(float(sy), 3) if sy is not None else ''))
                tracks = updated_tracks
                for tid in list(histories):
                    if tid not in tracks:
                        del histories[tid]
                text(panel, f'Detections: {len(detections)}',
                     (20, size[1] - 55), .6, (240, 240, 240))
                text(panel, 'Stability assumes a stationary washer',
                     (20, size[1] - 25), .42, (175, 190, 200))

            combined = np.hstack([display, panel])
            cv2.imshow(window, combined)
            key = cv2.waitKey(30 if cap is not None else 80) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                calibrating = True
                corners = []
                print('Select TL, TR, BR, BL corners in the image area')
            elif key == ord('r'):
                corners = []
            elif key in (10, 13) and calibrating:
                if len(corners) != 4:
                    print('Need four corner clicks')
                else:
                    try:
                        H = calibrate(corners, args.width_mm, args.height_mm)
                        save_calibration(calpath, H, size,
                                         (args.width_mm, args.height_mm), corners)
                        calibrating = False
                        tracks.clear()
                        histories.clear()
                        print('Calibration saved:', calpath)
                    except ValueError as exc:
                        print('Calibration failed:', exc)
            elif key == ord('s'):
                if H is None:
                    print('Calibrate before saving real-world coordinates')
                elif not last_rows:
                    print('No detections to save')
                else:
                    out = Path(args.csv)
                    exists = out.exists() and out.stat().st_size > 0
                    with out.open('a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=list(last_rows[0]))
                        if not exists:
                            writer.writeheader()
                        writer.writerows(last_rows)
                    print('Saved', len(last_rows), 'detections to', out)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
