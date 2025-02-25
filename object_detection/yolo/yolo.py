"""
This module provides functions for object detection using the YOLO model.
It includes methods for detecting objects in images and displaying the results using OpenCV.
"""

import numpy as np
import urllib
import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO(r"E:\GitHub\DeepLearning\runs\detect\train2\weights\best.pt")

def detect_objects_v1(img, classes=[], conf=0.5):
    """
    Detect objects in an image using the YOLO model.

    Args:
        img (str or np.ndarray): The image or image path to detect objects in.
        classes (list, optional): List of class IDs to filter detections. Defaults to [].
        conf (float, optional): Confidence threshold for detections. Defaults to 0.5.

    Returns:
        list: List of detection results.
    """
    if classes:
        return model.predict(img, classes=classes, conf=conf)
    return model.predict(img, conf=conf)

def predict_and_detect_using_cv2(img_url=None, img_path=None, classes=[], conf=0.5):
    """
    Predict and detect objects in an image from a URL using the YOLO model and display the results using OpenCV.

    Args:
        img_url (str): URL of the image to detect objects in.
        classes (list, optional): List of class IDs to filter detections. Defaults to [].
        conf (float, optional): Confidence threshold for detections. Defaults to 0.5.
    """
    if img_url:
        req = urllib.request.urlopen(img_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)  # 'Load it as it is'
    else:
        image = cv2.imread(img_path)

    results = detect_objects_v1(img_url, classes=classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(image, result.names[int(box.cls[0])], (int(box.xyxy[0][0]), int(box.xyxy[0][1])-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)

    while True:
        # Use the function cv2.imshow() to show an image in a window.
        # The window automatically fits to the image size.
        # First argument is the window name.
        # Second argument is the image to be displayed.
        # Each created window should have different window names.
        # Show original image:
        
        cv2.imshow("YoloPredict", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

def detect_objects_v2(image):
    """
    Detect objects in an image using the YOLO model and return the detections.

    Args:
        image (str or np.ndarray): The image or image path to detect objects in.

    Returns:
        list: List of detections with bounding box coordinates, confidence, and class name.
    """
    # Perform object detection on an image using the model
    results = model(image)[0]

    detections = []
    # Extract the bounding box coordinates, confidence, and class id
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        detections.append([int(x1), int(y1), int(x2), int(y2), round(score, 3),
                           results.names[int(class_id)]])
    
    # Visualize results on the frame
    # annotated_frame = results.plot()
    # cv2.imshow("window", annotated_frame)
    results.show()
    return detections

# Example usage
predict_and_detect_using_cv2(img_path=r"E:\GitHub\DeepLearning\parasites-2\test\images\Trichuris-Trichiura--259-_jpg.rf.a9b4d3caafeaebcdb18cdfdd81d6edbf.jpg")
# predict_and_detect_using_cv2("https://ultralytics.com/images/bus.jpg")

