import cv2
import time
import numpy as np
import mediapipe as mp
import math
import screen_brightness_control as sbc

################################
wCam, hCam = 640, 480
################################
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = mp.solutions.hands.Hands(None, 2,
        0.7, 0.5)
detector = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
minBright = 0
maxBright = 100
bright = 0
brightBar = 100
brightPer = 0

def findHands(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms,mp.solutions.hands.HAND_CONNECTIONS)
        return img, results
        
def findPosition(img, results, handNo=0, draw=True):
    xList = []
    yList = []
    bbox = []
    lmList = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            # print(id, cx, cy)
            lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
            (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

    return lmList, bbox
            
while (cv2.waitKey(1) == -1):
    success, img = cap.read()
    findHandsResult = findHands(img)
    if findHandsResult:
        img, results = findHandsResult
        lmList = findPosition(img, results, draw=False)
       
        if len(lmList) != 0 and len(lmList[0]) != 0:
            x1, y1 = lmList[0][4][1], lmList[0][4][2]
            x2, y2 = lmList[0][8][1], lmList[0][8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
      
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            # Hand range 50 - 300
            # Brightness Range 0 - 100
            bright = np.interp(length, [9, 125], [minBright, maxBright])
            brightBar = np.interp(length, [9, 125], [400, 150])
            brightPer = np.interp(length, [9, 125], [0, 100])
            sbc.set_brightness(str(bright))
            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(brightBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(brightPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        if img is not None:
            cv2.imshow("Img", img)