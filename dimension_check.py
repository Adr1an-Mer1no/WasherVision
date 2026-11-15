"""Experimental OD/ID measurement for flat, isolated washers; no automatic class override."""
import cv2
import numpy as np
from vision_utils import WASHERS, pixel_to_workspace


def _world_diameter(points, H):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    center = world.mean(axis=0)
    radius = np.median(np.linalg.norm(world-center, axis=1))
    return 2 * float(radius)


def measure_washer(frame, box, H):
    """Return OD, ID, center, quality, contour (or None fields).

    Contour-based dark-ring segmentation. Rejects ambiguous geometry rather than
    forcing a diameter. Requires dark washers on a brighter background.
    """
    if H is None:
        return dict(od=None, id=None, center=None, quality='UNCALIBRATED', contour=None, inner_contour=None)
    height, width = frame.shape[:2]
    x1,y1,x2,y2 = map(float, box)
    bw,bh=x2-x1,y2-y1
    if min(bw,bh)<15:
        return dict(od=None,id=None,center=None,quality='TOO_SMALL',contour=None,inner_contour=None)
    margin=int(max(bw,bh)*0.16)
    xa=max(0,int(x1)-margin); ya=max(0,int(y1)-margin)
    xb=min(width,int(x2)+margin); yb=min(height,int(y2)+margin)
    gray=cv2.cvtColor(frame[ya:yb,xa:xb],cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    # Otsu threshold assumes dark washers and a light background.
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return dict(od=None,id=None,center=None,quality='NO_CONTOUR',contour=None,inner_contour=None)
    bx=(x1+x2)/2-xa; by=(y1+y2)/2-ya
    candidates=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area<30 or len(c)<5: continue
        (cx,cy),radius=cv2.minEnclosingCircle(c)
        if radius<5: continue
        if abs(cx-bx)>bw*.35 or abs(cy-by)>bh*.35: continue
        # Outer edge should roughly correspond to the detection box.
        if not (.28*min(bw,bh) <= 2*radius <= 1.55*max(bw,bh)): continue
        perimeter=cv2.arcLength(c,True)
        circularity=4*np.pi*area/(perimeter*perimeter) if perimeter else 0
        candidates.append((abs(2*radius-max(bw,bh))/max(bw,bh) +
                           .4*np.hypot(cx-bx,cy-by)/max(bw,bh),i,c,circularity))
    if not candidates:
        return dict(od=None,id=None,center=None,quality='NO_OUTER_EDGE',contour=None,inner_contour=None)
    _,idx,outer,circularity=min(candidates,key=lambda x:x[0])
    # Ellipse fitting compensates for perspective by sampling its perimeter.
    if circularity < .55:
        return dict(od=None,id=None,center=None,quality='IRREGULAR_OUTER_EDGE',contour=None,inner_contour=None)
    ellipse=cv2.fitEllipse(outer)
    (cx,cy),(major,minor),angle=ellipse
    if min(major,minor)<8 or max(major,minor)/min(major,minor)>1.7:
        return dict(od=None,id=None,center=None,quality='ELLIPSE_UNRELIABLE',contour=None,inner_contour=None)
    theta=np.linspace(0,2*np.pi,72,endpoint=False)
    a=np.deg2rad(angle)
    xx=(major/2)*np.cos(theta); yy=(minor/2)*np.sin(theta)
    px=cx+xx*np.cos(a)-yy*np.sin(a)+xa
    py=cy+xx*np.sin(a)+yy*np.cos(a)+ya
    od=_world_diameter(np.column_stack((px,py)),H)
    if not 7 <= od <= 30:
        return dict(od=None,id=None,center=None,quality='OD_OUT_OF_RANGE',contour=None,inner_contour=None)
    # Identify a child contour representing the central opening.
    inner_candidates=[]
    h=hierarchy[0]
    child=int(h[idx][2])
    while child!=-1:
        inner=contours[child]
        if len(inner)>=5:
            (ix,iy),(imaj,imin),iang=cv2.fitEllipse(inner)
            if (np.hypot(ix-cx,iy-cy)<max(major,minor)*.15 and
                .20 < max(imaj,imin)/max(major,minor) < .85 and
                min(imaj,imin)>5):
                inner_candidates.append((cv2.contourArea(inner),inner))
        child=int(h[child][0])
    inner_d=None
    inner_contour=None
    if inner_candidates:
        inner=max(inner_candidates,key=lambda z:z[0])[1]
        (ix,iy),(imaj,imin),iang=cv2.fitEllipse(inner)
        a=np.deg2rad(iang)
        xx=imaj/2*np.cos(theta); yy=imin/2*np.sin(theta)
        pts=np.column_stack((ix+xx*np.cos(a)-yy*np.sin(a)+xa,
                             iy+xx*np.sin(a)+yy*np.cos(a)+ya))
        inner_d=_world_diameter(pts,H)
        if not (2 <= inner_d < od*.92):
            inner_d=None
        else:
            inner_contour=inner+np.array([[[xa,ya]]],dtype=np.int32)
    # A fitted outer edge much larger than the YOLO box is likely a shadow,
    # background feature or merged contour; reject instead of reporting a size.
    contour_box=cv2.boundingRect(outer)
    if contour_box[2] > bw*1.20 or contour_box[3] > bh*1.20:
        return dict(od=None,id=inner_d,center=(cx+xa,cy+ya),
                    quality='OUTER_EDGE_EXCEEDS_BOX',contour=outer+np.array([[[xa,ya]]],dtype=np.int32),
                    inner_contour=inner_contour)
    return dict(od=od,id=inner_d,center=(cx+xa,cy+ya),
                quality='OD_ID_MEASURED' if inner_d is not None else 'ID_UNAVAILABLE',
                contour=outer+np.array([[[xa,ya]]],dtype=np.int32),
                inner_contour=inner_contour)


def dimension_recommendation(od, inner_d, yolo_name, od_tolerance=2.0, id_tolerance=1.5):
    """Compare known sizes; recommendation is advisory, never a motor authorization."""
    if od is None:
        return 'NO_RELIABLE_MEASUREMENT', None
    candidates=[]
    for name,spec in WASHERS.items():
        od_error=abs(spec['od_mm']-od)
        id_error=abs(spec['id_mm']-inner_d) if inner_d is not None else 0
        if od_error<=od_tolerance and (inner_d is None or id_error<=id_tolerance):
            candidates.append((od_error + id_error, name))
    if not candidates:
        return 'SIZE_OUT_OF_RANGE', None
    best=min(x[0] for x in candidates)
    near=sorted(name for error,name in candidates if error<=best+.6)
    if yolo_name in near:
        return 'SIZE_CONSISTENT' if inner_d is not None else 'OD_ONLY_CONSISTENT',yolo_name
    return 'REVIEW_SIZE_MISMATCH','/'.join(near)
