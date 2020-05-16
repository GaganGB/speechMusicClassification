from numpy import mean
from numpy import std
import numpy as np
import shutil
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
import multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("\n"*25)

def roll_mean(X, wind_size=5, wind_jump=3):
    X = np.asarray(X)
    i = 0
    Y = []
    for i in range(0, X.shape[1], i+wind_jump):
        Y.append(np.mean(X[:,i:i+wind_size],axis=1))
    Y = np.transpose(np.asarray(Y))
    return np.nan_to_num(Y)

def roll_std(X, wind_size=5, wind_jump=3):
    X = np.asarray(X)
    i = 0
    Y = []
    for i in range(0, X.shape[1], i+wind_jump):
        Y.append(np.std(X[:,i:i+wind_size],axis=1))
    Y = np.transpose(np.asarray(Y))
    return np.nan_to_num(Y)

def roll_variationFromMean(X, wind_size=5, wind_jump=3):
    X = np.asarray(X)
    i = 0
    Y = []
    for i in range(0, X.shape[1], i+wind_jump):
        Y.append(stats.variation(X[:,i:i+wind_size],axis=1))
    Y = np.transpose(np.asarray(Y))
    return np.nan_to_num(Y)

def create_dataset_train(filename):
    wave = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
    # rms_feature = feature.rms(y=wave, frame_length=256, hop_length=128)
    # zero_cross_rate = feature.zero_crossing_rate(y=wave, frame_length=256, hop_length=128)
    # spec_centroid = feature.spectral_centroid(y=wave, sr=22050, n_fft=256, hop_length=128)
    # chroma = feature.chroma_stft(y=wave, sr=22050, n_fft=256, hop_length=128)
    # spec_flat = feature.spectral_flatness(y=wave, n_fft=256, hop_length=128)
    # poly_feat = feature.poly_features(y=wave, sr=22050, n_fft=256, hop_length=128)
    mfcc = feature.mfcc(y=wave, sr=22050, n_mfcc=15, n_fft=256, hop_length=128)
    # final = np.concatenate((rms_feature, zero_cross_rate, spec_centroid, chroma, spec_flat, poly_feat))
    final = mfcc
    final = np.concatenate((roll_mean(final),roll_std(final),roll_variationFromMean(final)), axis=0)
    np.savetxt(fname = "../data/lstm_feature/train/" + str(filename[:-4]) + ".txt", X = final)
    print(filename)
    return 0

def create_dataset_test(filename):
    wave = np.loadtxt(fname = "../data/normalized_data/test/" + str(filename))
    # rms_feature = feature.rms(y=wave, frame_length=256, hop_length=128)
    # zero_cross_rate = feature.zero_crossing_rate(y=wave, frame_length=256, hop_length=128)
    # spec_centroid = feature.spectral_centroid(y=wave, sr=22050, n_fft=256, hop_length=128)
    # chroma = feature.chroma_stft(y=wave, sr=22050, n_fft=256, hop_length=128)
    # spec_flat = feature.spectral_flatness(y=wave, n_fft=256, hop_length=128)
    # poly_feat = feature.poly_features(y=wave, sr=22050, n_fft=256, hop_length=128)
    mfcc = feature.mfcc(y=wave, sr=22050, n_mfcc=15, n_fft=256, hop_length=128)
    # final = np.concatenate((rms_feature, zero_cross_rate, spec_centroid, chroma, spec_flat, poly_feat))
    final = mfcc
    final = np.concatenate((roll_mean(final),roll_std(final),roll_variationFromMean(final)), axis=0)
    np.savetxt(fname = "../data/lstm_feature/test/" + str(filename[:-4]) + ".txt", X = final)
    print(filename)
    return 0

def create_dataset():
    if(os.path.exists("../data/lstm_feature") == True):
        try:
           shutil.rmtree("../data/lstm_feature/train")
           shutil.rmtree("../data/lstm_feature/test")
        except:
           print('Error while deleting directory')
    os.mkdir("../data/lstm_feature/train")
    os.mkdir("../data/lstm_feature/test")
    pool = mp.Pool(mp.cpu_count())
    list_dir = sorted(os.listdir("../data/normalized_data/train"))
    pool.map(create_dataset_train, [name for name in list_dir])
    list_dir = sorted(os.listdir("../data/normalized_data/test"))
    pool.map(create_dataset_test, [name for name in list_dir])
    pool.close()

def load_dataset_train(filename):
    trainX = np.transpose(np.loadtxt(fname = "../data/lstm_feature/train/" + str(filename)))
    return trainX

def load_dataset_test(filename):
    testX = np.transpose(np.loadtxt(fname = "../data/lstm_feature/test/" + str(filename)))
    return testX

def load_train_label(filename):
    trainy = [1 if (filename[0:1]=='M') else 0, 1 if (filename[0:1]=='S') else 0]
    return trainy

def load_test_label(filename):
    testy = [1 if (filename[0:1]=='M') else 0, 1 if (filename[0:1]=='S') else 0]
    return testy

def load_dataset():
    pool = mp.Pool(mp.cpu_count())
    list_dir = sorted(os.listdir("../data/lstm_feature/train"))
    trainX = pool.map(load_dataset_train, [name for name in list_dir])
    trainy = pool.map(load_train_label, [name for name in list_dir])
    list_dir = sorted(os.listdir("../data/lstm_feature/test"))
    testX = pool.map(load_dataset_test, [name for name in list_dir])
    testy = pool.map(load_test_label, [name for name in list_dir])
    pool.close()
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
    verbose, epochs, batch_size = 2, 15, 256
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(hidden_layer, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_layer, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Starting training")
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    print("Starting Testing")
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
    predictions = model.predict(testX)
    print(predictions)
    return accuracy

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def run_experiment(repeats=3):
    trainX, trainy, testX, testy = load_dataset()
    repeats = 10
    r = 0
    scores = []
    hid = 400
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy, hid)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    print("hid:" + str(hid))
    summarize_results(scores)

# create_dataset()
run_experiment()

print("Done")
