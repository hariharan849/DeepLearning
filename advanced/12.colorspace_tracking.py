import cv2 
import numpy as np 
 
# Capture the input frame from webcam 
def get_frame(cap, scaling_factor): 
    # Capture the frame from video capture object 
    ret, frame = cap.read() 
 
    # Resize the input frame 
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA) 
 
    return frame 

if __name__=='__main__': 
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
    scaling_factor = 0.5 
 
    # Define 'red' range in HSV colorspace 
    lower1_red = np.array([0,50,50]) 
    upper1_red = np.array([100,255,255]) 
    
    lower2_red = np.array([170,50,50]) 
    upper2_red = np.array([180,255,255]) 

    while True: 
        frame = get_frame(cap, scaling_factor) 
 
        # Convert the HSV colorspace 
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
 
        # Threshold the HSV image to get only blue color 
        mask1 = cv2.inRange(hsv_frame, lower1_red, upper1_red) 
        
        mask2 = cv2.inRange(hsv_frame, lower2_red, upper2_red) 
        
        mask = mask1 + mask2
 
        # Bitwise-AND mask and original image 
        res = cv2.bitwise_and(frame, frame, mask=mask) 
        res = cv2.medianBlur(res, ksize=5) 
 
        cv2.imshow('Original image', frame) 
        cv2.imshow('Color Detector', res) 
 
        # Check if the user pressed ESC key 
        c = cv2.waitKey(delay=10) 
        if c == 27: 
            break 
    
    cap.release()
    cv2.destroyAllWindows() 