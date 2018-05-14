from skimage.transform import pyramid_gaussian
import argparse 
import cv2

for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
    if resized.shape[0]< 24 or resized.shape[1] < 24:
        break

    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # normalize window
        #net2 = P2()
        #net2.train()
        
