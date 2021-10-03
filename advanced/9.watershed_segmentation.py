import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# Naive approach
sep_coins = cv2.imread('../DATA/pennies.jpg')

sep_blur = cv2.medianBlur(sep_coins,65)
cv2_imshow(sep_blur)

gray_sep_coins = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_sep_coins)

ret, sep_thresh = cv2.threshold(gray_sep_coins,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2_imshow(sep_thresh)

contours, hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# For every entry in contours
for i in range(len(contours)):
    
    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1:
        
        # We can now draw the external contours from the list of contours
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
		


# water shed algorithm
#Step 1: Read Image
img = cv2.imread('/content/sample_data/separate_coins.jpg')
#Step 2: Apply Blur
img = cv2.medianBlur(img,65)
#Step 3: Convert to Grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Step 4: Apply Threshold (Inverse Binary with OTSU as well)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2_imshow(thresh)
#Optional Step 5: Noise Removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#Step 6: Grab Background that you are sure of
sure_bg = cv2.dilate(opening,kernel,iterations=3)
#Step 7: Find Sure Foreground
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

#Step 8: Find Unknown Region
#Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Step 9: Label Markers of Sure Foreground
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
#Step 10: Apply Watershed Algorithm to find Markers
markers = cv2.watershed(img,markers)
#Step 11: Find Contours on Markers
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# For every entry in contours
for i in range(len(contours)):
    
    # last column in the array is -1 if an external contour (no contours inside of it)
    # if hierarchy[0][i][3] == -1:
        
        # We can now draw the external contours from the list of contours
    cv2.drawContours(img, contours, i, (255, 0, 0), 10)

cv2_imshow(img)