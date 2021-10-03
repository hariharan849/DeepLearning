import cv2

videoCapture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'MyOutputVid.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    #MJPG, XVID

while True:
    # Capture frame-by-frame
    success, frame = videoCapture.read()
    
    if success: # Loop until there are no more frames.
        videoWriter.write(frame)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
videoCapture.release()
cv2.destroyAllWindows()