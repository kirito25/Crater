from skimage.transform import pyramid_gaussian
import argparse
import cv2
import sys, time
import keras_cnn
import numpy as np
from helpers import *


image_1 = cv2.imread("../../crater_dataset/crater_data/images/tile3_24.pgm")
image_2 = cv2.imread("../../crater_dataset/crater_data/images/tile3_25.pgm")
(winW, winH) = (100, 100)
window_size = (winH, winW)

images = [image_1] #, image_2]

def crater_sliding_window(image, stepSize):
    for image_layer in pyramid(image, scale=2):
        for (x, y, window) in sliding_window(image_layer, stepSize, window_size):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            yield (x, y, window, image_layer)

model = keras_cnn.network()

for (x, y, window, image_layer) in crater_sliding_window(image_1, 30):
    if np.argmax(model.predict(keras_cnn.cvImg2input(window))) == 1:
        cv2.rectangle(image_layer, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    copy = image_layer.copy()
    cv2.rectangle(copy, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("window", copy)
    cv2.waitKey(1)
    time.sleep(0.025)

