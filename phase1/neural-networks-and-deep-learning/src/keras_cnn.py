"""
Dependencies are: keras, matplotlib, and numpy
Do a the following installations:
    pip install matplotlib
    pip install tensorflow tensorflow-gpu
    pip install keras
    pip install numpy

Add the following to your ~/.bashrc
"""
import keras, os, crater_loader
import keras.backend as K
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
# stop showing the gpu info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

num_classes = 2 # number of output neuron
batch_size = 200
epochs = 25
# input image dimensions
img_x, img_y = 200, 200
input_shape = (img_x, img_y, 1)

# takes an image and returns it in a format to be
# fed to the keras model
def cvImg2input(img):
    return img.reshape(1, img_x, img_y, 1)

# custom logger
class Logs(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, epoch, logs={}):
        pass

#custom metrics
def num_of_craters(y_true, y_pred):
    y_true = K.argmax(y_true, 1)
    return tf.count_nonzero(y_true)

def num_of_non_craters(y_true, y_pred):
    y_true = K.argmax(y_true, 1)
    y_true = tf.cast(y_true, bool)
    y_true = tf.logical_not(y_true)
    return tf.count_nonzero(y_true)

def tp(y_true, y_pred):
    y_true = K.argmax(y_true, 1)
    y_pred = K.argmax(y_pred, 1)
    correct_answers = K.equal(y_true, y_pred)
    crater = tf.cast(y_true, bool)
    tp = tf.logical_and(correct_answers, crater)
    return tf.count_nonzero(tp)

def tn(y_true, y_pred):
    y_true = K.argmax(y_true, 1)
    y_pred = K.argmax(y_pred, 1)
    correct_answers = K.equal(y_true, y_pred)
    crater = tf.cast(y_true, bool)
    tp = tf.logical_and(correct_answers, crater)
    tn = tf.logical_not(tp)
    return tf.count_nonzero(tn)

def fp(y_true, y_pred):
    y_true = tf.cast(K.argmax(y_true), bool)
    y_pred = tf.cast(K.argmax(y_pred), bool)
    t = tf.logical_not(y_true)
    fp = tf.logical_and(t, y_pred)
    return tf.count_nonzero(fp)

def fn(y_true, y_pred):
    y_true = tf.cast(K.argmax(y_true), bool)
    y_pred = tf.cast(K.argmax(y_pred), bool)
    t = tf.logical_not(y_pred)
    fn = tf.logical_and(t, y_true)
    return tf.count_nonzero(fn)

# Creating plots from the metrics
def plot(title="", data=None, filename=None, xlabel="", ylabel=""):
    if data == None or filename == None:
        return
    if type(data[0]) == list:
        for l in data:
            plt.plot(range(1, epochs + 1), l)
    else: 
        plt.plot(range(1, epochs + 1), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(np.arange(1, epochs + 1, 5.0))
    plt.savefig(filename)
    plt.close()

def network():
    """
    Returns a keras network already trained
    """
    logger = Logs()

    training_data, validation_data, test_data = crater_loader.load_crater_data_phaseII_wrapper()
    training_data_x , training_data_y = crater_loader.reformat(training_data, False)
    validation_data_x , validation_data_y = crater_loader.reformat(validation_data, False)
    test_data_x , test_data_y = crater_loader.reformat(test_data, False)

    training_data_x = training_data_x.reshape(training_data_x.shape[0], img_x, img_y, 1)
    validation_data_x = validation_data_x.reshape(validation_data_x.shape[0], img_x, img_y, 1)
    test_data_x = test_data_x.reshape(test_data_x.shape[0], img_x, img_y, 1)

    test_data_y = keras.utils.to_categorical(test_data_y, num_classes)
    validation_data_y = keras.utils.to_categorical(validation_data_y, num_classes)
    training_data_y = keras.utils.to_categorical(training_data_y, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd',  metrics=['accuracy', tp, tn, fn, fp, num_of_non_craters, num_of_craters])

    history = model.fit(training_data_x, training_data_y, batch_size=batch_size, epochs=epochs, 
            callbacks=[logger], validation_data=(validation_data_x, validation_data_y))

    print "\nEvaluating on test data"
    score = model.evaluate(test_data_x, test_data_y, verbose=1)
    print 'Test loss: %.2f' %  ( float(score[0]) )
    print 'Test accuracy: %.2f' % ( float(score[1]) )
    print "Creating plots ..."
    # Creates the plot 
    plot(data=history.history['acc'], title="Accuracy vs Epochs on training data", 
            filename='training_accuracy_vs_epochs', xlabel="Epoch", ylabel="Accuracy")
    plot(data=history.history['val_acc'], title="Accuracy vs Epochs on validation data", 
            filename='validation_accuracy_vs_epochs', xlabel="Epoch", ylabel="Accuracy")
   
    plot(data=history.history['fn'], title="FN vs Epoch on training data", filename='training_fn_vs_epoch', xlabel="Epoch", ylabel="FN")
    plot(data=history.history['fp'], title="FP vs Epoch on training data", filename='training_fp_vs_epoch', xlabel="Epoch", ylabel="FP")
    
    plot(data=[history.history['tp'], history.history['num_of_craters']], title="TP vs Epoch training data", 
            filename='training_tp_vs_epoch', xlabel="Epoch", ylabel="TP")
    plot(data=[history.history['tn'], history.history['num_of_non_craters']], title="TN vs Epoch on training data", 
            filename='training_tn_vs_epoch', xlabel="Epoch", ylabel="TN")

    detection_rate = np.array(history.history['tp']) / ( np.array(history.history['tp']) + np.array(history.history['fn']) )
    plot(data=detection_rate.tolist(), title="Detection rate vs Epochs on training data", 
            filename='training_detection_rate_vs_epochs', xlabel="Epoch", ylabel="Detection Rate")
    
    false_rate = np.array(history.history['fp']) / ( np.array(history.history['tp']) + np.array(history.history['fp']) )
    plot(data=false_rate.tolist(), title="False rate vs Epochs on training data", 
            filename='training_false_rate_vs_epochs', xlabel="Epoch", ylabel="False Rate")
    
    quality_rate = np.array(history.history['tp']) / ( np.array(history.history['tp']) + np.array(history.history['fn']) + np.array(history.history['fp']) )
    plot(data=quality_rate.tolist(), title="Quality rate vs Epochs on training data", 
            filename='training_quality_rate_vs_epochs', xlabel="Epoch", ylabel="Quality Rate")
    return model

if __name__ == '__main__':
    network()

