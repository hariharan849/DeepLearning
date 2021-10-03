"""
Example to introduce how to read a video file backwards
"""

# Import the required packages:
# from google.colab.patches import cv2_imshow
import cv2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capture = cv2.VideoCapture(r"D:\hariharan\DeepLearning\openCV\datasets\test1.mp4")

# Check if camera opened successfully:
if capture.isOpened()is False:
    print("Error opening video stream or file")

# We get the index of the last frame of the video file:
frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
print("starting in frame: '{}'".format(frame_index))

# Read until video is completed:
while frame_index >= 0:

    # We set the current frame position:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Capture frame-by-frame from the video file:
    ret, frame = capture.read()

    if ret is True:

        # Print current frame number per iteration
        # print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))

        # Get the timestamp of the current frame in milliseconds
        # print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))

        # Display the resulting frame:
        cv2.imshow("image",frame)

        # Decrement the index to read next frame:
        frame_index = frame_index - 1
        print("next index to read: '{}'".format(frame_index))
 
        # Press q on keyboard to exit the program:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything:
capture.release()
cv2.destroyAllWindows()
