import os

import cv2
import numpy

def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # The file cannot be loaded as an image.
                    # Skip it.
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)
    return names, training_images, training_labels
    
face_cascade = cv2.CascadeClassifier(
    'D:/hariharan/DeepLearning/openCV/advanced/haarcascade_frontalface_default.xml')
path_to_training_images = 'D:/hariharan/DeepLearning/openCV/advanced/custom_face_recognization/data'
training_image_size = (200, 200)
names, training_images, training_labels = read_images(
    path_to_training_images, training_image_size)

"""
LBPH instead divides a detected face into small cells
and, for each cell, builds a histogram that describes
whether the brightness of the image is increasing when
comparing neighboring pixels in a given direction.
This cell's histogram can be compared to the
corresponding cell's in the model, producing a measure
of similarity.
"""
"""
radius: The pixel distance between the neighbors that are used to calculate a cell's histogram (by default, 1)
neighbors: The number of neighbors used to calculate a cell's histogram (by default, 8)
grid_x: The number of cells into which the face is divided horizontally (by default, 8)
grid_y: The number of cells into which the face is divided vertically (by default, 8)
confidence: The confidence threshold (by default, the highest possible floating-point value so that no results are discarded)
"""
model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_images, training_labels)

camera = cv2.VideoCapture(0)
while (cv2.waitKey(1) == -1):
    success, frame = camera.read()
    if success:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[x:x+w, y:y+h]
            if roi_gray.size == 0:
                # The ROI is empty. Maybe the face is at the image edge.
                # Skip it.
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)
            text = '%s, confidence=%.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', frame)