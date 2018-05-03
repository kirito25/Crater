"""
crater_loader
~~~~~~~~~~~~
"""

#### Libraries
import os
import cv2 as cv
import random
import numpy as np

def load_data(folder = ''):
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
    SRC = "../../crater_dataset/crater_data/images/normalize_images/" + folder

    i = 0
    # assume its a non-crater
    VALUE = 0
    if folder == "crater":
        VALUE = 1
    for filename in os.listdir(SRC):
        img = cv.imread(SRC + "/" + filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.flatten() / 255.0
        img = img.reshape(len(img),1)
        images.append((img, SRC + "/" + filename))
        labels.append(vectorized_result(VALUE))

    data = zip(images, labels)
    random.shuffle(data)
    return data

def load_data_wrapper(split=[0.8,0.2]):
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
    crater_data    = load_data("crater")
    noncrater_data = load_data("non-crater")
    
    n_crater_data    = len(crater_data)
    n_noncrater_data = len(noncrater_data)

    train_end       = validation_start = int(split[0] * n_crater_data)
    validation_end  = test_start       = int(split[1] * n_crater_data) + train_end

    training_data = crater_data[:train_end] 
    validation    = crater_data[validation_start:validation_end] 
    test          = crater_data[test_start:] 
    
    train_end       = validation_start = int(split[0] * n_noncrater_data)
    validation_end  = test_start = int(split[1] * n_noncrater_data) + train_end
    training_data  += noncrater_data[:train_end]
    validation     += noncrater_data[validation_start:validation_end]
    test           += noncrater_data[test_start:]

    random.shuffle(training_data)
    random.shuffle(validation)
    random.shuffle(test)
    
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

def load_crater_data_phaseII_wrapper():
    # 70% training_data , 15% validation data , and remaining test data
    return load_data_wrapper([0.70, 0.15])
