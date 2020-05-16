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
from librosa import util
from librosa import feature
from scipy import stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def normalize():
#     for _,filename in enumerate(sorted(os.listdir("../data/normalized_data/train"))):
#         wave = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
#         final = util.normalize(wave)
#         np.savetxt(fname = "../data/lstm_file/train/" + str(filename[:-4]) + ".txt", X = final)
#         print(filename)
#     for _,filename in enumerate(sorted(os.listdir("wav/test"))):
#         wave = np.loadtxt(fname = "wav/test/" + str(filename))
#         final = util.normalize(wave)
#         np.savetxt(fname = "lstm_file/test/" + str(filename[:-4]) + ".txt", X = final)
#         print(filename)

def roll_mean(X, wind_size=5, wind_jump=3):
    X = np.asarray(X)
    n = 0
    i = 0
    Y = []
    Z = []
    W = []
    for i in range(0, X.shape[1], i+wind_jump):
        # print(X[:,i:i+wind_size])
        Y.append(np.mean(X[:,i:i+wind_size],axis=1))
        Z.append(np.std(X[:,i:i+wind_size],axis=1))
        W.append(stats.variation(X[:,i:i+wind_size],axis=1))
    Y = np.transpose(np.asarray(Y))
    Z = np.transpose(np.asarray(Z))
    W = np.transpose(np.asarray(W))
    final = np.nan_to_num(np.concatenate((Y, Z, W), axis = 0))
    return final

def create_dataset():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for _,filename in enumerate(sorted(os.listdir("../data/normalized_data/train"))):
        wave = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
        rms_feature = feature.rms(y=wave, frame_length=256, hop_length=128)
        zero_cross_rate = feature.zero_crossing_rate(y=wave, frame_length=256, hop_length=128)
        # spec_centroid = feature.spectral_centroid(y=wave, sr=22050, n_fft=256, hop_length=128)
        # chroma = feature.chroma_stft(y=wave, sr=22050, n_fft=256, hop_length=128)
        # spec_flat = feature.spectral_flatness(y=wave, n_fft=256, hop_length=128)
        # poly_feat = feature.poly_features(y=wave, sr=22050, n_fft=256, hop_length=128)
        # mfcc = feature.mfcc(y=wave, sr=22050, n_mfcc=15, n_fft=256, hop_length=128)
        # final = np.concatenate((rms_feature, zero_cross_rate, spec_centroid, chroma, spec_flat, poly_feat))
        final = np.concatenate((rms_feature, zero_cross_rate))
        final = roll_mean(final)
        np.savetxt(fname = "../data/lstm_feature/train/" + str(filename[:-4]) + ".txt", X = final)
        Y_train.append(filename)
        print(filename)
    for _,filename in enumerate(sorted(os.listdir("../data/normalized_data/test"))):
        wave = np.loadtxt(fname = "../data/normalized_data/test/" + str(filename))
        rms_feature = feature.rms(y=wave, frame_length=256, hop_length=128)
        zero_cross_rate = feature.zero_crossing_rate(y=wave, frame_length=256, hop_length=128)
        # spec_centroid = feature.spectral_centroid(y=wave, sr=22050, n_fft=256, hop_length=128)
        # chroma = feature.chroma_stft(y=wave, sr=22050, n_fft=256, hop_length=128)
        # spec_flat = feature.spectral_flatness(y=wave, n_fft=256, hop_length=128)
        # poly_feat = feature.poly_features(y=wave, sr=22050, n_fft=256, hop_length=128)
        # mfcc = feature.mfcc(y=wave, sr=22050, n_mfcc=15, n_fft=256, hop_length=128)
        # final = np.concatenate((rms_feature, zero_cross_rate, spec_centroid, chroma, spec_flat, poly_feat))
        final = np.concatenate((rms_feature, zero_cross_rate))
        final = roll_mean(final)
        np.savetxt(fname = "../data/lstm_feature/test/" + str(filename[:-4]) + ".txt", X = final)
        Y_test.append(filename)
        print(filename)



def load_dataset():
    list_dir = sorted(os.listdir("../data/lstm_feature/train"))
    list_dir_test = sorted(os.listdir("../data/lstm_feature/test"))
    trainX = []
    trainy = []
    testX = []
    testy = []
    count = 0
    for _,filename in enumerate(list_dir):
        trainX.append(np.transpose(np.loadtxt(fname = "../data/lstm_feature/train/" + str(filename))))
        trainy.append([1 if (filename[0:1]=='M') else 0, 1 if (filename[0:1]=='S') else 0])
    count = 0
    for _,filename in enumerate(list_dir_test):
        testX.append(np.transpose(np.loadtxt(fname = "../data/lstm_feature/test/" + str(filename))))
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
    model.add(Dropout(0.5))
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
def run_experiment(repeats=3):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    repeats = 10
    r = 0
    scores = []
    hid = 385
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy, hid)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    print("hid:" + str(hid))
    summarize_results(scores)

def run_experiment1():
    load_dataset()

create_dataset()
run_experiment()
# 280-72, 295-73, 310-75, 340-70, 385-77, 430-70, 445-75, 460-72

print("Done")
