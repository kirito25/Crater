"""
crater_loader
~~~~~~~~~~~~
"""

#### Libraries
import os
import cv2 as cv
import random
import numpy as np

def load_data():
    """
    This fuction will return a list of tuples (x, y).
    x is an nparray as a column vector of the image flatten 
    and y is 0 or 1 depending if x is crater or not.

    Returns
    -------
        [(x1, y1), ..., (xn, yn)] : xi is a column vector of
                                    an image in gray scale.
                                    yi is a column vector corresponding
                                    to the desired output neurons.
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

    data = zip(images, labels)
    random.shuffle(data)
    return data

def load_data_wrapper():
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)``. 

    In particular, ``training_data`` is a list containing
    2-tuples ``(x, y)`` in the same format as load_data().

    The validation_data and test_data are in the format of 
    a list of tuple (x, y). x is an image be the input to the 
    neural network. y is the index of which output neuron should be
    activated.

    """
    data = load_data()

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

