from skimage.transform import pyramid_gaussian
import argparse 
import cv2
from crater_loader import *
from crater_slice_window import *
from random import randint
import keras
from keras.models import model_from_json

winW, winH = 50,50
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
SRC = "/home/user11/Project/Crater/phase1/crater_dataset/crater_data/images/tile3_24/crater/"

def sliding_wind(img, stepSize=1, windowSize=(2, 2)):
    n = img.shape[0]
    for i in range(n-windowSize[0]):
        for j in range(n-windowSize[1]):
            slide = img[i:i+windowSize[0],j:j+windowSize[1]]

#dp = np.array([[randint(0, 9) for i in range(0, 10)] for j in range(0, 10)])
#sliding_window(dp, 1, (2, 2))

json_p2 = open('model2.json', 'r')
json_model2 = json_p2.read()
json_p2.close()
model2 = model_from_json(json_model2)
model2.load_weights('model2.h5')
cnt = 0
for filename in os.listdir(SRC):
    image = cv2.imread(SRC + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
        if resized.shape[0]< 50 or resized.shape[1] < 50:
            break

        best_img = None
        best_window = None
        best_window_p = 0
        for (x, y, window) in sliding_window(resized, stepSize=20, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # normalize window
            cv2.normalize(window, window, 0, 255, cv2.NORM_MINMAX)
            window = window[...,np.newaxis]
            window = np.expand_dims(window, axis=0)
            window = window.reshape(1, 50, 50, 1)
            prediction = model2.predict(window)
            if prediction[0][0] > best_window_p and cnt < 10:
                best_img = image.copy()
                best_window = (x, y)
                best_window_p = prediction[0][0]
        cv2.rectangle(best_img, (best_window[0], best_window[1]),
                (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imwrite('./detected'+str(20+cnt) +'.jpg', best_img)
        cnt += 1
        print prediction
    print "Done with one image, moving to the next"
    print "-------------------------------------------------------------"
            #net2 = P2()
            #net2.train()
            
