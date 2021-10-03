import dlib, cv2
import numpy as np

dat = r"D:\hariharan\opencv\Computer-Vision-with-Python\Face\shape_predictions\shape_predictor_68_face_landmarks.dat"

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat)


while True: 
    # read the input frame 
    ret, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
        print (rect)
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


    cv2.imshow("Output", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()