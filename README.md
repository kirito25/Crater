# Crater Detection

Simple experiment using Neural Neural networks
to detect craters in Mars.
To prepare the training data, do the following:
```
$ cd dataset
$ unzip crater_data.zip
$ python preprocess.py
```

The preprocess.py script can accept some parameters, 
you can see them by doing:
```
$ python preprocess.py -h
```

There are two main experiments:
- Using a simple neural network (crater_network.py)
- Using a Convolution network using Keras (keras_cnn.py)

Almost all the programs accept a `-h` flag to show possible options.
Feel free to mess with them

These can be run after preprocessing the data.
Depending on the parameters you have chosen to pre-process
the data, you might have to modify some variables
in keras_cnn.py. (they will be at the top of the file)

To run the simple experiment do the following:
```
$ cd src
$ python run_experiment.py
```

To run using the convolutional network, do:
```
$ cd src
$ python keras_cnn.py
```

The convolution network run will create some graphs
about its detection, accuracy, etc.
It will also save the trained network for future use.

