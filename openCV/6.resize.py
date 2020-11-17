"""
OpenCV read tutorial
"""
from google.colab.patches import cv2_imshow
import cv2

# Use the function cv2.imread() to read an image.
# The image should be in the working directory or a full path of image should be provided.
# load OpenCV logo image:
# Params: (image, flag)
#    image(str): Path of the image
#    flag(int): 0(cv2.IMREAD_GRAYSCALE), 1(cv2.IMREAD_COLOR) or -1(cv2.IMREAD_UNCHANGED)
image = cv2.imread("logo.png")

# Resizing the channels
# Interpolations
# 1. cv2.INTER_AREA
# 2. cv2.INTER_CUBIC
# 3. cv2.INTER_LINEAR
# 4. cv2.INTER_NEAREST
# 5. cv2.INTER_LANCZOS4
image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_LINEAR)
#image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)


# Use the function cv2.imshow() to show an image in a window.
# The window automatically fits to the image size.
# First argument is the window name.
# Second argument is the image to be displayed.
# Each created window should have different window names.
# Show original image:
cv2_imshow(image)
# cv2.imshow("OpenCV logo", image)

# cv2.waitKey() is a keyboard binding function.
# The argument is the time in milliseconds.
# The function waits for specified milliseconds for any keyboard event.
# If any key is pressed in that time, the program continues.
# If 0 is passed, it waits indefinitely for a key stroke.
# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(0)

# To destroy all the windows we created call cv2.destroyAllWindows()
cv2.destroyAllWindows()