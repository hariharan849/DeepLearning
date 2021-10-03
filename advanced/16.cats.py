import cv2
import numpy as np

model = r"C:\Users\Win7\Anaconda\Lib\site-packages\cv2\data\haarcascade_frontalcatface.xml"
img_path = r"D:\hariharan\opencv\Computer-Vision-with-Python\Face\cats_detection_opencv\cats.jpg"

img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detect = cv2.CascadeClassifier(model)
rects = detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(150, 150))

for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

while True:
    cv2.imshow("Output", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()