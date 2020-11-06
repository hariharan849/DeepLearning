from keras.preprocessing.image import image_to_array

class ImageToArrayPreprocessor(object):
    def __init__(self, data_format):
        self.data_format = data_format

    def preprocess(self, image):
        return image_to_array(image, data_format=self.data_format)