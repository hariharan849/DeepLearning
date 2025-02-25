from ultralytics import YOLO
import cv2
import numpy as np

results = {}

#load the model
coco_model = YOLO('yolov11n.pt')
license_plate_detector = YOLO('yolov11n.pt')

cap = cv2.VideoCapture('./sample.mp4')


# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
