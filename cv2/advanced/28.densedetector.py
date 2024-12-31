import cv2
import numpy as np

class DenseDetector(object):
    def __init__(self, step_size=20, feature_scale=40,
    img_bound=20):
        # Create a dense feature detector
        self.detector = cv2.FeatureDetector_create("Dense")
        # Initialize it with all the required parameters
        self.detector.setInt("initXyStep", step_size)
        self.detector.setInt("initFeatureScale", feature_scale)
        self.detector.setInt("initImgBound", img_bound)

    def detect(self, img):
        # Run feature detector on the input image
        return self.detector.detect(img)
        
if __name__=='__main__':
    input_image = cv2.imread("../img.jpg")
    input_image_sift = np.copy(input_image)
    # Convert to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints = DenseDetector(20,20,5).detect(input_image)
    # Draw keypoints on top of the input image
    input_image = cv2.drawKeypoints(input_image, keypoints,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the output image
    cv2.imshow('Dense feature detector', input_image)
    # Initialize SIFT object
    sift = cv2.SIFT()
    # Detect keypoints using SIFT
    keypoints = sift.detect(gray_image, None)
    # Draw SIFT keypoints on the input image
    input_image_sift = cv2.drawKeypoints(input_image_sift,
    keypoints,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the output image
    cv2.imshow('SIFT detector', input_image_sift)
    # Wait until user presses a key
    cv2.waitKey()