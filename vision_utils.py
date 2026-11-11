"""Pure geometry and conservative overlap checks for the washer vision system."""
import json
from pathlib import Path
import cv2
import numpy as np

WASHERS = {
    'N1': {'material': 'nylon', 'od_mm': 20, 'id_mm': 10},
    'W1': {'material': 'metal', 'od_mm': 11, 'id_mm': 8},
    'W2': {'material': 'metal', 'od_mm': 18, 'id_mm': 11},
    'W3': {'material': 'metal', 'od_mm': 20, 'id_mm': 10},
    'W4': {'material': 'metal', 'od_mm': 22, 'id_mm': 7},
}

def box_centroid(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return ((x1+x2)/2, (y1+y2)/2)

def calibrate(image_points, width_mm, height_mm):
    pts = np.asarray(image_points, dtype=np.float32)
    if pts.shape != (4,2) or not np.isfinite(pts).all():
        raise ValueError('Exactly four finite pixel corners are required')
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError('Target dimensions must be positive')
    if abs(cv2.contourArea(pts)) < 100:
        raise ValueError('Corners are too close together')
    target = np.float32([[0,0],[width_mm,0],[width_mm,height_mm],[0,height_mm]])
    H = cv2.getPerspectiveTransform(pts, target)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Invalid homography')
    return H

def pixel_to_workspace(cx, cy, H):
    p = np.float32([[[cx,cy]]])
    x,y = cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64))[0,0]
    return float(x),float(y)

def bbox_iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter else 0

def overlaps(box, boxes, idx, threshold=0.08):
    # Only a screening flag: boxes cannot determine whether washers are stacked.
    return any(j!=idx and bbox_iou(box,other)>=threshold for j,other in enumerate(boxes))

def save_calibration(path, H, image_size, dimensions, points):
    data={'homography':np.asarray(H).tolist(), 'image_size':list(image_size),
          'target_mm':list(dimensions), 'image_points':np.asarray(points).tolist()}
    Path(path).write_text(json.dumps(data,indent=2))

def load_calibration(path, image_size):
    data=json.loads(Path(path).read_text())
    if list(image_size)!=data['image_size']:
        raise ValueError('Camera resolution changed; recalibrate')
    H=np.asarray(data['homography'],dtype=np.float64)
    if H.shape!=(3,3) or not np.isfinite(H).all():
        raise ValueError('Invalid saved calibration')
    return H
