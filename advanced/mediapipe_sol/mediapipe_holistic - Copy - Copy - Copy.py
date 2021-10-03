#imports
import os
import numpy as np
import csv
import mediapipe as mp
import cv2

#mediapipe drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#read from camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#initialize holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened(): 
        # read the input frame 
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        #make detections
        results = holistic.process(image)
        
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 121), thickness=2, circle_radius=2)
        )
        
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        cv2.imshow("Output", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()