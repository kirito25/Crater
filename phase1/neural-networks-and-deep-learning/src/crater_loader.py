"""
crater_loader
~~~~~~~~~~~~
"""

#### Libraries
import os
import cv2 as cv
import random
import numpy as np

def load_data(debug = False):
    """
    This fuction will return a list of tuples (x, y).
    x is an nparray of the image flatten and y is 0 or 1
    depending if x is crater, 1, or not, 0.
    """
    
    images = []
    labels = []
    SRC = "../../crater_dataset/crater_data/images/normalize_images/"

    i = 0
    for directory in os.listdir(SRC):
        # assume its a non-crater
        VALUE = 0
        if directory == "crater":
            VALUE = 1
        for filename in os.listdir(SRC + directory):
            img = cv.imread(SRC + directory + "/" + filename)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = img.flatten() / 255.0
            img = img.reshape(200*200,1)
            images.append(img)
            labels.append(vectorized_result(VALUE))
            if debug:
                i += 1
                if i > 10:
                    break
        if debug:
            break

    data = zip(images, labels)
    random.shuffle(data)
    return data

def load_data_wrapper(debug = False):
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)``. 

    In particular, ``training_data`` is a list containing
    2-tuples ``(x, y)``.  ``x`` is a 200*200-dimensional numpy.ndarray
    containing the input image.  ``y`` is 1 dimensional vector 0 or 1.
    1 being a crater and 0 not being a crater.
    """
    data = load_data(debug)

    training_data = data[:500]
    validation = data[500:800]
    test = data[800:1000]
    
    test_data = []
    validation_data = []

    for (x, y) in test:
        test_data.append((x, np.argmax(y)))
    for (x, y) in validation:
        validation_data.append((x, np.argmax(y)))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((2,1), dtype=int)
    e[j] = 1
    return e

