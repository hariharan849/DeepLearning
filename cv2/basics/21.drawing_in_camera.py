import cv2
# Create a function based on a CV2 Event (Left button click)

# mouse callback function
def draw_rectangle(event, x, y, flags, param):
    """ Draw a rectangle on the image
    Args:
        event: The event that was triggered
        x: The x coordinate of the event
        y: The y coordinate of the event
        flags: Any flags that were passed
        param: Any parameters that were passed
    """
    global pt1,pt2,topLeft_clicked,botRight_clicked

    # handle only for left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset the rectangle if both points are selected
        if topLeft_clicked == True and botRight_clicked == True:
            topLeft_clicked = False
            botRight_clicked = False
            pt1 = (0, 0)
            pt2 = (0, 0)

        # set the top left point
        if topLeft_clicked == False:
            pt1 = (x,y)
            topLeft_clicked = True
        
        # set the bottom right point
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True

        
# Haven't drawn anything yet!

pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
botRight_clicked = False

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

# Create a named window for connections
cv2.namedWindow('Test')

# Bind draw_rectangle function to mouse cliks
cv2.setMouseCallback('Test', draw_rectangle) 

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if topLeft_clicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0,0,255), thickness=-1)
        
    #drawing rectangle
    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Test', frame)

    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()