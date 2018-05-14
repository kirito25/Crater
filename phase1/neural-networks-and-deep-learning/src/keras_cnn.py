"""
Dependencies are: keras, matplotlib, and numpy
Do a the following installations:
    pip install matplotlib
    pip install tensorflow tensorflow-gpu
    pip install keras
    pip install numpy

Add the following to your ~/.bashrc
"""
import keras, os
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import crater_loader
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# stop showing the gpu info
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Logs(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.acc = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))

num_classes = 2
batch_size = 200
epochs = 20
# input image dimensions
img_x, img_y = 200, 200
input_shape = (img_x, img_y, 1)
logs = Logs()

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

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])
model.fit(training_data_x, training_data_y, batch_size=batch_size, epochs=epochs, 
        callbacks=[logs], validation_data=(validation_data_x, validation_data_y))

print "\nEvaluating on test data"
score = model.evaluate(test_data_x, test_data_y, verbose=1)
print 'Test loss: %.2f' % ( float(score[0]) )
print 'Test accuracy: %.2f' % ( float(score[1]) )

# Creates the plot of "Accuracy vs Epochs during training"
plt.plot(range(1,epochs + 1), logs.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy vs Epochs during training")
plt.xticks(np.arange(1, epochs, 1.0))
plt.savefig('training_accuracy_vs_epochs')




