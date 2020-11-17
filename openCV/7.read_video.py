"""
Example to introduce how to read a video file
"""

# Import the required packages
from google.colab.patches import cv2_imshow
import cv2

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture("test.avi")
 
# Check if the video is opened successfully
if capture.isOpened()is False:
    print("Error opening the video file!")
 
# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret is True:
        # Display the resulting frame
        cv2_imshow(frame)
 
        # Press q on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
