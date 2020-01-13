import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution1D

'http://www.jussihuotari.com/2017/12/20/spell-out-convolution-1d-in-cnns/'

def Conv1D_numpy():
    conv1d_filter = np.array([1,2])
    data = np.array([0, 3, 4, 5])
    result = []
    for i in range(3):
        print(data[i:i+2], "*", conv1d_filter, "=", data[i:i+2] * conv1d_filter)
        result.append(np.sum(data[i:i+2] * conv1d_filter))
        print("Conv1d output", result)


def Conv1D_Keras():
    K.clear_session()
    toyX = np.array([0, 3, 4, 5]).reshape(1,4,1)
    toyY = np.array([6, 11, 14]).reshape(1,3,1)

    toy = Sequential([
          Convolution1D(filters=1, kernel_size=2, strides=1, padding='valid',
                        use_bias=False, input_shape=(4,1), name='Conv1D')])
    toy.compile(optimizer=Adam(lr=5e-2), loss='mae')
    print("Initial random guess conv weights", toy.layers[0].get_weights()[0].reshape(2,))

    for i in range(200):
        h = toy.fit(toyX, toyY, verbose=0)
        if i%20 == 0:
           print("{:3d} {} \t {}".format(i, toy.layers[0].get_weights()[0][:,0,0], h.history))

def Conv1D_and_Channels():
    K.clear_session()
    toyX = np.array([[0, 0], [3, 6], [4, 7], [5, 8]]).reshape(1,4,2)
    toyy = np.array([30, 57, 67]).reshape(1,3,1)
    toy = Sequential([
            Convolution1D(filters=1, kernel_size=2, strides=1, padding='valid',
                          use_bias=False, input_shape=(4,2), name='Conv1D')])
    toy.compile(optimizer=Adam(lr=5e-2), loss='mae')
    print("Initial random guess conv weights", toy.layers[0].get_weights()[0].reshape(4,))

    # Expecting [1, 3], [2, 4]
    for i in range(200):
        h = toy.fit(toyX, toyy, verbose=0)
        if i%20 == 0:
           print("{:3d} {} \t {}".format(i, toy.layers[0].get_weights()[0].reshape(4,), h.history))

def Conv1D_and_Multiple_Filters():
    K.clear_session()
    toyX = np.array([0, 3, 4, 5]).reshape(1,4,1)
    toyy = np.array([[6, 12], [11, 25], [14, 32]]).reshape(1,3,2)
    toy = Sequential([
            Convolution1D(filters=2, kernel_size=2, strides=1, padding='valid',
                          use_bias=False, input_shape=(4,1), name='Conv1D')])
    toy.compile(optimizer=Adam(lr=5e-2), loss='mae')
    print("Initial random guess conv weights", toy.layers[0].get_weights()[0].reshape(4,))

    for i in range(200):
        h = toy.fit(toyX, toyy, verbose=0)
        if i%20 == 0:
           print("{:3d} {} \t {}".format(i, toy.layers[0].get_weights()[0][:,0,0], h.history))

    # Feature 2 weights should be 3 and 4
    toy.layers[0].get_weights()[0][:,0,1]
