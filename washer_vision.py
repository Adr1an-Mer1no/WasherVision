"""Live YOLO11 washer detection + planar calibration. No motor commands are sent."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import cv2
from ultralytics import YOLO
from vision_utils import (WASHERS, box_centroid, calibrate, pixel_to_workspace,
                          overlaps, save_calibration, load_calibration)

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Washer vision and XY coordinate logger')
    p.add_argument('--model',default=str(ROOT/'models'/'best.pt'))
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--confidence',type=float,default=0.5)
    p.add_argument('--width-mm',type=float,default=200)
    p.add_argument('--height-mm',type=float,default=150)
    p.add_argument('--calibration',default=str(ROOT/'calibration.json'))
    p.add_argument('--csv',default=str(ROOT/'washer_coordinates.csv'))
    p.add_argument('--image',default=None,help='Optional image instead of webcam (for testing)')
    args=p.parse_args()
    if not Path(args.model).exists():
        p.error(f'Model missing: {args.model}')
    model=YOLO(args.model)
    cap=None
    if args.image:
        source=cv2.imread(args.image)
        if source is None: p.error('Could not open image')
    else:
        cap=cv2.VideoCapture(args.camera)
        if not cap.isOpened(): p.error(f'Cannot open camera {args.camera}')
        ok,source=cap.read()
        if not ok: p.error('Cannot read webcam')
    size=(source.shape[1],source.shape[0])
    H=None
    calpath=Path(args.calibration)
    if calpath.exists():
        try:
            H=load_calibration(calpath,size)
            print('Loaded calibration:',calpath)
        except (ValueError,KeyError) as exc:
            print('Calibration not used:',exc)
    corners=[]
    calibrating=False
    last_rows=[]
    window='WasherVision | C calibrate | ENTER confirm | R reset | S save | Q quit'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    def click(event,x,y,flags,param):
        if calibrating and event==cv2.EVENT_LBUTTONDOWN and len(corners)<4:
            corners.append((x,y))
    cv2.setMouseCallback(window,click)
    try:
        while True:
            if cap:
                ok,frame=cap.read()
                if not ok: break
            else: frame=source.copy()
            if (frame.shape[1],frame.shape[0])!=size:
                raise RuntimeError('Image resolution changed; restart and recalibrate')
            display=frame.copy()
            last_rows=[]
            if calibrating:
                for i,(x,y) in enumerate(corners):
                    cv2.circle(display,(x,y),5,(0,140,255),-1)
                    cv2.putText(display,str(i+1),(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,140,255),2)
                cv2.putText(display,'Click TL, TR, BR, BL of physical calibration rectangle',
                            (12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            else:
                result=model.predict(frame,conf=args.confidence,verbose=False)[0]
                if result.boxes is not None:
                    boxes=result.boxes.xyxy.cpu().numpy().tolist()
                    clsids=result.boxes.cls.cpu().numpy().astype(int).tolist()
                    scores=result.boxes.conf.cpu().numpy().tolist()
                    for i,(box,clsid,score) in enumerate(zip(boxes,clsids,scores)):
                        name=str(model.names[clsid])
                        cx,cy=box_centroid(box)
                        # A detection bounding box is not proof of a clear/flat washer.
                        overlap=overlaps(box,boxes,i)
                        status='REVIEW_OVERLAP' if overlap else 'UNVERIFIED_FLAT'
                        x_mm=y_mm=None
                        if H is not None:
                            x_mm,y_mm=pixel_to_workspace(cx,cy,H)
                        x1,y1,x2,y2=map(int,box)
                        color=(0,165,255) if overlap else (0,210,80)
                        cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
                        cv2.circle(display,(round(cx),round(cy)),4,(0,0,255),-1)
                        xy=f'X:{x_mm:.1f} Y:{y_mm:.1f} mm' if H is not None else 'UNCALIBRATED'
                        label=f'{name} {score:.2f} {xy} {status}'
                        cv2.putText(display,label,(max(0,x1),max(18,y1-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX,.46,color,2)
                        spec=WASHERS.get(name,{})
                        last_rows.append(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            class_name=name,confidence=round(float(score),4),
                            pixel_x=round(cx,2),pixel_y=round(cy,2),
                            workspace_x_mm=round(x_mm,2) if x_mm is not None else '',
                            workspace_y_mm=round(y_mm,2) if y_mm is not None else '',
                            material=spec.get('material','unknown'),
                            od_mm=spec.get('od_mm',''),id_mm=spec.get('id_mm',''),
                            status=status, motor_command_sent=False))
                cv2.putText(display,'CALIBRATED' if H is not None else 'NOT CALIBRATED - press C',
                            (10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
            cv2.imshow(window,display)
            key=cv2.waitKey(30 if cap else 80)&0xFF
            if key in (ord('q'),27): break
            if key==ord('c'):
                calibrating=True; corners=[]
                print('Select corners TL, TR, BR, BL following the physical target axes')
            elif key==ord('r'):
                corners=[]
            elif key in (10,13) and calibrating:
                if len(corners)!=4:
                    print('Need four corner clicks')
                else:
                    try:
                        H=calibrate(corners,args.width_mm,args.height_mm)
                        save_calibration(calpath,H,size,(args.width_mm,args.height_mm),corners)
                        calibrating=False
                        print('Calibration saved:',calpath)
                    except ValueError as exc: print('Calibration failed:',exc)
            elif key==ord('s'):
                if H is None:
                    print('Calibrate before saving real-world coordinates')
                elif not last_rows:
                    print('No detections to save')
                else:
                    out=Path(args.csv)
                    exists=out.exists() and out.stat().st_size>0
                    with out.open('a',newline='',encoding='utf-8') as f:
                        writer=csv.DictWriter(f,fieldnames=list(last_rows[0]))
                        if not exists: writer.writeheader()
                        writer.writerows(last_rows)
                    print('Saved',len(last_rows),'detections to',out)
    finally:
        if cap: cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
