import cv2
import numpy as np

# Create a black image.
img = np.zeros((800, 800, 3), np.uint8)

# Initialize the Kalman filter.
kalman = cv2.KalmanFilter(4, 2)
# The state is (x, y, dx, dy), where (x, y) is the position and (dx, dy) is the velocity.
kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]], np.float32)
# Transition matrix is 4x4 because we have 4 state variables.
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32)
# Process noise covariance is 4x4 because we have 4 state variables.
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], np.float32) * 0.03

last_measurement = None
last_prediction = None

# Callback function for mouse events.
def on_mouse_moved(event, x, y, flags, param):
    global img, kalman, last_measurement, last_prediction
    
    # Draw a white circle at the current mouse position.
    measurement = np.array([[x], [y]], np.float32)
    if last_measurement is None:
        # This is the first measurement.
        # Update the Kalman filter's state to match the measurement.
        kalman.statePre = np.array(
            [[x], [y], [0], [0]], np.float32)
        kalman.statePost = np.array(
            [[x], [y], [0], [0]], np.float32)
        prediction = measurement
    else:
        # This is not the first measurement.
        kalman.correct(measurement)
        # Predict the next state.
        prediction = kalman.predict()  # Gets a reference, not a copy

        # Trace the path of the measurement in green.
        cv2.line(img, (last_measurement[0], last_measurement[1]),
                 (measurement[0], measurement[1]), (0, 255, 0))

        # Trace the path of the prediction in red.
        cv2.line(img, (last_prediction[0], last_prediction[1]),
                 (prediction[0], prediction[1]), (0, 0, 255))

    last_prediction = prediction.copy()
    last_measurement = measurement

cv2.namedWindow('kalman_tracker')
cv2.setMouseCallback('kalman_tracker', on_mouse_moved)

while True:
    cv2.imshow('kalman_tracker', img)
    k = cv2.waitKey(1)
    if k == 27:  # Escape
        #cv2.imwrite('kalman.png', img)
        break
