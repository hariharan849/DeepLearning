import numpy as np
import cv2
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

img = cv2.imread('sample_data/00-puppy.jpg')
img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
print (img.shape)
# create a mask
mask = np.zeros(img.shape[:2],np.uint8)

# specify the background and foreground model
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# specify the region of interest
rect =  (50, 30, 500, 600)
# apply grabcut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# create a mask where the background is 0
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
# apply the mask to the image
img = img*mask2[:,:,np.newaxis]

cv2_imshow(img)