import cv2

class SimpleProcessor(object):
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        return cv2.resize(image, (width, height), interpolation=self.interpolation)