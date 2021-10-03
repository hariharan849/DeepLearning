import cv2
import numpy as np

model = r"C:\Users\Win7\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
#img_path = r"D:\hariharan\opencv\Computer-Vision-with-Python\Face\cats_detection_opencv\cats.jpg"

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

detect = cv2.CascadeClassifier(model)

while True: 
    # read the input frame 
    ret, img = cap.read() 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #preprocess : aligning
    rects = detect.detectMultiScale(gray)

    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow("Output", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()