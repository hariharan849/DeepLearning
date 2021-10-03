#imports
import os
import numpy as np
import csv
import mediapipe as mp
import cv2

#mediapipe drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection

#read from camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#initialize holistic
with mp_face.FaceDetection(min_detection_confidence=0.5) as face:
    while cap.isOpened(): 
        # read the input frame 
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        #make detections
        results = face.process(image)
        
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        cv2.imshow("Output", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()