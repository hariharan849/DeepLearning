import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab
 
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
 
count = 0
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    count += 1
    cv2.imwrite("hari{0}.jpg".format(count), imgS)
    
    cv2.imshow('Webcam',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count > 10:
        break