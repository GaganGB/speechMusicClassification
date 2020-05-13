from numpy import mean
from numpy import std
from numpy import dstack
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_dataset1():
    list_dir = sorted(os.listdir("data/train/features"))
    list_dir_test = sorted(os.listdir("data/test/features"))
    trainX = []
    trainy = []
    testX = []
    testy = []
    count = 0
    for _,filename in enumerate(list_dir):
        trainX.append(np.transpose(np.loadtxt(fname = "data/train/features/" + str(filename))))
        trainy.append([1 if (filename[0:1]=='M') else 0, 1 if (filename[0:1]=='S') else 0])
    count = 0
    for _,filename in enumerate(list_dir_test):
        testX.append(np.transpose(np.loadtxt(fname = "data/test/features/" + str(filename))))
        testy.append([1 if (filename[0:1]=='M') else 0, 1 if (filename[0:1]=='S') else 0])
    trainy = np.asarray(trainy)
    trainX = np.asarray(trainX)
    testy = np.asarray(testy)
    testX = np.asarray(testX)
    print(trainX.shape)
    print(trainy.shape)
    print(testX.shape)
    print(testy.shape)
    return trainX, trainy, testX, testy


def evaluate_model(trainX, trainy, testX, testy, hidden_layer):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(hidden_layer, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_layer, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    hid = 210
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy, hid)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    print("hid:" + str(hid))
    summarize_results(scores)
# run the experiment
run_experiment()


print("Done")