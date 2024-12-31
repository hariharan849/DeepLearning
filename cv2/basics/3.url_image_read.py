"""
OpenCV url read tutorial
"""
import cv2
import urllib.request
import numpy as np

req = urllib.request.urlopen('https://variety.com/wp-content/uploads/2021/12/doctor-strange.jpg?w=681&h=383&crop=1&resize=681%2C383')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1) # 'Load it as it is'

while True:
    # Use the function cv2.imshow() to show an image in a window.
    # The window automatically fits to the image size.
    # First argument is the window name.
    # Second argument is the image to be displayed.
    # Each created window should have different window names.
    # Show original image:
    
    cv2.imshow("image", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
        
# To destroy all the windows we created call cv2.destroyAllWindows()
cv2.destroyAllWindows()