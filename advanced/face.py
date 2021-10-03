import cv2
import numpy as np

proto = r"D:\hariharan\opencv\Computer-Vision-with-Python\Face\study\deploy.prototxt.txt"
caffe = r'D:\hariharan\opencv\Computer-Vision-with-Python\Face\study\res10_300x300_ssd_iter_140000.caffemodel'
face_threshold = 0.50
caffe_model = cv2.dnn.readNetFromCaffe(proto, caffe)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

while True: 
    # read the input frame 
    ret, img = cap.read() 
    (w, h) = img.shape[:2]
    #create blog image
    blob_img = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), -1, (300, 300), (104., 177., 123.))

    # Create dnn model
    caffe_model.setInput(blob_img)

    predictions = caffe_model.forward()

    for i in range(predictions.shape[2]):
        threshold = predictions[0, 0, i, 2]
        if threshold > face_threshold:
            coords = predictions[0, 0, i, 3:7] * np.array((w, h, w, h))
            (startX, startY, endX, endY) = coords.astype("int")

            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
    cv2.imshow("Output", img)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()