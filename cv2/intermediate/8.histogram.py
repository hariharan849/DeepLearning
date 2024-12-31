""" Histogram
"""
import cv2
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

#intensity distribution of an image.
#It is a plot with pixel values (ranging from 0 to 255, not always) in X-axis and corresponding number of pixels in the image on Y-axis.

#  contrast, brightness, intensity distribution

# Histogram Calculation
"""
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, “[img]”.
channels : it is also given in square brackets. It the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0],[1] or [2] to calculate histogram of blue,green or red channel respectively.
mask : mask image. To find histogram of full image, it is given as “None”. But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
ranges : this is our RANGE. Normally, it is [0,256].
"""

img = cv2.imread('sample_data/00-puppy.jpg')
color = ('b','g','r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0, 256])
plt.show()


img = cv2.imread('sample_data/00-puppy.jpg')
hist_values = cv2.calcHist([img],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.plot(histr,color = "b")
plt.xlim([0,256])
plt.show()

eq_gorilla = cv2.equalizeHist(img)
cv2_imshow(eq_gorilla)


hist_values = cv2.calcHist([eq_gorilla],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.plot(histr,color = "b")
plt.xlim([0,256])
plt.show()



#######################
# color image

color_gorilla = cv2.imread('../DATA/gorilla.jpg')
hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2_imshow(eq_gorilla)




######################
# clahe
img = cv2.imread('tsukuba_l.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)