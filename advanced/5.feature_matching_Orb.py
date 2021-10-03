import numpy as np
import cv2
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow

#reeses = cv2.imread(r'D:\hariharan\DeepLearning\openCV\datasets\lucky_charms.jpg')
#cereals = cv2.imread(r'D:\hariharan\DeepLearning\openCV\datasets\many_cereals.jpg')
reeses = cv2.imread(r'D:\hariharan\DeepLearning\openCV\advanced\nasa_logo.png')
cereals = cv2.imread(r'D:\hariharan\DeepLearning\openCV\advanced\kennedy_space_center.jpg')
cereals = cv2.resize(cereals, (600, 600), interpolation=cv2.INTER_LINEAR)

# Create SIFT Object
#sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(reeses,None)
kp2, des2 = sift.detectAndCompute(cereals,None)
# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#matches = bf.match(des1, des2)
# Sort the matches by distance.
#matches = sorted(matches, key=lambda x:x.distance)
# Draw the best 25 matches.
#sift_matches = cv2.drawMatches(
#    reeses, kp1, cereals, kp2, matches[:25], cereals,
#    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

#cv2.drawMatchesKnn expects list of lists as matches.
sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
while True:
    cv2.imshow("feature", sift_matches)
    c = cv2.waitKey(10) 
    if c == 27: 
        break 

