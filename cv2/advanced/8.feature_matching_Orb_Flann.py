import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

reeses = cv2.imread(r'/content/sample_data/nasa_logo.png')
cereals = cv2.imread(r'/content/sample_data/kennedy_space_center.jpg')
img0 = reeses
img1 = cereals

# Create SIFT Object
#sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp0, des0 = sift.detectAndCompute(img0,None)
kp1, des1 = sift.detectAndCompute(img1,None)

des0 = np.float32(des0)
des1 = np.float32(des1)
# Define FLANN-based matching parameters.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Perform FLANN-based matching.
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des0, des1, k=2)

# Prepare an empty mask to draw good matches.
mask_matches = [[0, 0] for i in range(len(matches))]

# Populate the mask based on David G. Lowe's ratio test.
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        mask_matches[i] = [1, 0]

# Draw the matches that passed the ratio test.
img_matches = cv2.drawMatchesKnn(
    img0, kp0, img1, kp1, matches, None,
    matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
    matchesMask=mask_matches, flags=0)

cv2_imshow(img_matches)
