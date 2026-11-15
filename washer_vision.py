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
from dimension_check import measure_washer, dimension_recommendation
from class_fusion import recommend, vote
from dimension_calibration import correct
from vision_utils import (WASHERS, box_centroid, calibrate, pixel_to_workspace,
                          overlaps, save_calibration, load_calibration)

ROOT = Path(__file__).resolve().parent
COLORS = {'high': (70, 205, 90), 'review': (0, 190, 255), 'low': (80, 100, 255)}


def text(img, message, pos, scale=.55, color=(235, 235, 235), thickness=1):
    cv2.putText(img, str(message), pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def assign_tracks(detections, tracks, next_id, max_distance=45):
    """Associate current detections to previous-frame centroids, by proximity.
    IDs are temporary; reappearing or crossing objects may receive new IDs.
    """
    candidates = []
    for i, det in enumerate(detections):
        for tid, track in tracks.items():
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
    parser.add_argument('--csv', default=str(ROOT / 'washer_coordinates_fusion.csv'))
    parser.add_argument('--image', default=None)
    parser.add_argument('--history', type=int, default=30)
    parser.add_argument('--vote-window', type=int, default=30)
    parser.add_argument('--vote-min', type=int, default=20)
    parser.add_argument('--vote-agreement', type=float, default=.80)
    parser.add_argument('--known-class', choices=list(WASHERS), default=None, help='For collecting labeled isolated washer measurements with K')
    parser.add_argument('--samples-csv', default=str(ROOT / 'dimension_samples.csv'))
    parser.add_argument('--dimension-calibration', default=str(ROOT / 'dimension_calibration.json'))
    parser.add_argument('--debug-edges', action='store_true', help='Show OD/ID contour legend')
    args = parser.parse_args()

    if not Path(args.model).exists():
        parser.error(f'Model missing: {args.model}')
    if args.vote_window < 2 or not (1 <= args.vote_min <= args.vote_window) or not (0 < args.vote_agreement <= 1):
        parser.error('Invalid voting parameters')
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

    dimension_model = None
    dimpath = Path(args.dimension_calibration)
    if dimpath.exists():
        import json
        try:
            dimension_model = json.loads(dimpath.read_text())
            if dimension_model.get('version') != 1:
                raise ValueError('Unsupported dimension calibration version')
            print('Loaded dimension correction:', dimpath)
        except (ValueError, KeyError) as exc:
            parser.error(f'Invalid dimension calibration: {exc}')
    raw_measurements = defaultdict(lambda: deque(maxlen=30))
    corners = []
    calibrating = False
    last_rows = []
    tracks = {}
    histories = defaultdict(lambda: deque(maxlen=args.history))
    class_histories = defaultdict(lambda: deque(maxlen=args.vote_window))
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
            if args.debug_edges:
                text(display, 'CYAN=outer  MAGENTA=hole  RED=YOLO center',
                     (15, size[1]-20), .65, (255, 255, 255), 2)
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
                    # Measurements are for flat, isolated washers only.
                    measurement = dict(od=None, id=None, center=None,
                                       quality='OVERLAP_OR_UNCALIBRATED', contour=None, inner_contour=None)
                    if not det['overlap'] and H is not None:
                        measurement = measure_washer(frame, det['box'], H)
                    diameter_mm = measurement['od']
                    inner_d_mm = measurement['id']
                    if measurement['quality'] == 'OD_ID_MEASURED' and diameter_mm is not None and inner_d_mm is not None:
                        raw_measurements[tid].append((diameter_mm, inner_d_mm))
                    corrected_od, corrected_id = correct(diameter_mm, inner_d_mm, dimension_model)
                    size_status, suggested = dimension_recommendation(
                        corrected_od, corrected_id, det['name'])
                    fusion = recommend(det['name'], det['score'], corrected_od,
                                       corrected_id, overlap=det['overlap'],
                                       measurement_quality=measurement['quality'])
                    class_histories[tid].append(fusion['suggested'])
                    final_class, vote_status, votes, vote_fraction = vote(
                        class_histories[tid], args.vote_window,
                        args.vote_min, args.vote_agreement)
                    score = det['score']
                    confidence_level = 'high' if score >= .85 else 'review' if score >= .65 else 'low'
                    status = ('REVIEW_OVERLAP' if det['overlap'] else
                              'REVIEW_CONFIDENCE' if confidence_level != 'high' else
                              'UNVERIFIED_FLAT')
                    if size_status in ('REVIEW_SIZE_MISMATCH', 'SIZE_OUT_OF_RANGE'):
                        status = size_status
                    color = COLORS['review'] if (det['overlap'] or size_status in ('REVIEW_SIZE_MISMATCH', 'SIZE_OUT_OF_RANGE')) else COLORS[confidence_level]
                    x1, y1, x2, y2 = map(int, det['box'])
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    if measurement['contour'] is not None:
                        cv2.drawContours(display, [measurement['contour']], -1,
                                         (255, 180, 0), 2)
                    # Cyan: estimated outer edge. Magenta: estimated inner hole.
                    if measurement.get('inner_contour') is not None:
                        cv2.drawContours(display, [measurement['inner_contour']], -1,
                                         (255, 0, 255), 2)
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
                    row_y = 165 + i * 115
                    if row_y + 95 < size[1]:
                        text(panel, f"#{tid} {det['name']}  {score:.2f}   {coords}",
                             (20, row_y), .56, color, 2)
                        od_text = f'OD: {diameter_mm:.1f}' if diameter_mm is not None else 'OD: ?'
                        od_text += f'  ID: {inner_d_mm:.1f} mm' if inner_d_mm is not None else '  ID: ?'
                        if dimension_model is not None and corrected_od is not None:
                            od_text += f'  corr: {corrected_od:.1f}/{corrected_id:.1f}'
                        if suggested and suggested != det['name']:
                            od_text += f'  possible: {suggested}'
                        text(panel, od_text, (20, row_y + 25), .45, (200, 220, 230))
                        advisory = final_class if final_class else '?'
                        text(panel, f'Suggest: {fusion["suggested"] or "?"}  Vote: {advisory} ({vote_fraction:.0%})',
                             (20, row_y + 43), .43, (150, 235, 235))
                        text(panel, fusion['status'] + ' | ' + vote_status,
                             (20, row_y + 61), .38, color)
                        # Show W2 versus W3 relative dimensional compatibility.
                        sc = fusion.get('scores', {})
                        if sc and row_y + 110 < size[1]:
                            text(panel, f'W2 score: {sc.get("W2", 0):.3f}  W3 score: {sc.get("W3", 0):.3f}',
                                 (20, row_y + 80), .39, (175, 220, 225))
                        if sx is not None:
                            text(panel, f'Stability ({len(history)}/{args.history}): '
                                 f'sX={sx:.2f}, sY={sy:.2f} mm',
                                 (20, row_y + 100), .38, (190, 205, 215))
                        else:
                            text(panel, f'Stability: {len(history)}/{args.history} frames',
                                 (20, row_y + 100), .38, (190, 205, 215))
                    spec = WASHERS.get(det['name'], {})
                    last_rows.append(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        class_name=det['name'], confidence=round(score, 4),
                        pixel_x=round(det['cx'], 2), pixel_y=round(det['cy'], 2),
                        workspace_x_mm=round(det['x_mm'], 2) if det['x_mm'] is not None else '',
                        workspace_y_mm=round(det['y_mm'], 2) if det['y_mm'] is not None else '',
                        material=spec.get('material', 'unknown'),
                        od_mm=spec.get('od_mm', ''), id_mm=spec.get('id_mm', ''),
                        status=status, measured_od_mm=round(diameter_mm, 2) if diameter_mm is not None else '',
                        measured_id_mm=round(inner_d_mm, 2) if inner_d_mm is not None else '',
                        corrected_od_mm=round(corrected_od, 2) if corrected_od is not None else '',
                        corrected_id_mm=round(corrected_id, 2) if corrected_id is not None else '',
                        measurement_quality=measurement['quality'],
                        dimension_status=size_status, dimension_suggestion=suggested or '',
                        fusion_suggestion=fusion['suggested'] or '', fusion_status=fusion['status'],
                        voted_class=final_class or '', vote_status=vote_status,
                        vote_support=round(vote_fraction, 3), motor_command_sent=False,
                        track_id=tid, stability_samples=len(history),
                        std_x_mm=round(float(sx), 3) if sx is not None else '',
                        std_y_mm=round(float(sy), 3) if sy is not None else ''))
                tracks = updated_tracks
                for tid in list(histories):
                    if tid not in tracks:
                        del histories[tid]
                        class_histories.pop(tid, None)
                        raw_measurements.pop(tid, None)
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
                        class_histories.clear()
                        raw_measurements.clear()
                        print('Calibration saved:', calpath)
                    except ValueError as exc:
                        print('Calibration failed:', exc)
            elif key == ord('k'):
                if not args.known_class:
                    print('For known-class sample collection, restart with --known-class W2 (or W1/W3/W4/N1).')
                elif H is None or len(detections) != 1:
                    print('Sample rejected: need calibration and EXACTLY ONE visible washer.')
                else:
                    tid = detections[0]['track_id']
                    vals = list(raw_measurements[tid])
                    if len(vals) < 30:
                        print(f'Sample rejected: {len(vals)}/30 valid OD/ID frames. Wait longer.')
                    else:
                        import time
                        med = np.median(np.array(vals), axis=0)
                        samplefile = Path(args.samples_csv)
                        exists = samplefile.exists() and samplefile.stat().st_size > 0
                        with samplefile.open('a', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=['class_name','raw_od_mm','raw_id_mm','timestamp_utc'])
                            if not exists: writer.writeheader()
                            writer.writerow({'class_name':args.known_class, 'raw_od_mm':round(float(med[0]),4),
                              'raw_id_mm':round(float(med[1]),4), 'timestamp_utc':datetime.now(timezone.utc).isoformat()})
                        raw_measurements[tid].clear()
                        print(f'Saved labeled {args.known_class}: OD={med[0]:.2f}, ID={med[1]:.2f} mm -> {samplefile}')
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
