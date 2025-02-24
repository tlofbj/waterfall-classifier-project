import os
import cv2
import numpy as np
    

def load_data(data_dir):

    signal_dir = os.path.join(data_dir, "signal")
    no_signal_dir = os.path.join(data_dir, "no_signal")

    X = [] # Feature vectors (flattened images), X is a matrix
    y = [] # Labels (1 for signal, 0 for no_signal), Y is a vector

    for filename in os.listdir(signal_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(signal_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            flattened_image = image.flatten()
            X.append(flattened_image)
            y.append(1)  # Label for signal

    for filename in os.listdir(no_signal_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(no_signal_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            flattened_image = image.flatten()
            X.append(flattened_image)
            y.append(0)  # Label for no_signal

    return np.array(X), np.array(y)

