import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(0, len(series) - window_size):
        X.append(series[i:window_size+i])
        y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units = 5, input_shape=(window_size,1)))
    model.add(Dense(1, activation='tanh'))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    return ''.join(i for i in text if i in ' abcdefghijklmnopqrstuvwxyz!,.:;?')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    steps = len(text) // window_size
    for i in range(0, steps):
        inputs.append(text[i*step_size:window_size+i*step_size])
        outputs.append(text[i*step_size+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units = 200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Dense(num_chars, activation='softmax'))
    return model
