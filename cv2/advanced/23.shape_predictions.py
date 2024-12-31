import dlib, cv2
import numpy as np

# Path to the shape predictor model
dat = r"..\shape_predictions\shape_predictor_68_face_landmarks.dat"

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it to the format (x, y, w, h) as we would normally do with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
# The shape predictor is used to predict the facial landmarks
predictor = dlib.shape_predictor(dat)


while True: 
    # read the input frame 
    ret, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
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