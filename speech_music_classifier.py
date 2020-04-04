# Importing librosa to extract feartures from the audio files

import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import joblib


train_dir = "wav/train"
test_dir = "wav/test"
def create_dirs():
    if(os.path.exists("data") == False):
        os.mkdir("data")
    if(os.path.exists("data/train") == False):
        os.mkdir("data/train")
    if(os.path.exists("data/test") == False):
        os.mkdir("data/test")
    if(os.path.exists("data/train/features") == False):
        os.mkdir("data/train/features")
    if(os.path.exists("data/test/features") == False):
        os.mkdir("data/test/features")
    if(os.path.exists("data/train/kmean_features") == False):
        os.mkdir("data/train/kmean_features")
    if(os.path.exists("data/test/kmean_features") == False):
        os.mkdir("data/test/kmean_features")
    print("Done Directory creation")

def feature_extraction_train(hop_length):
    list_dir = sorted(os.listdir(train_dir))
    for _, filename in enumerate(list_dir):
        y, sr = librosa.load("wav/train/" + str(filename))
        mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length))
        centroid = np.array(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length))
        zero_crossing_rate = np.array(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length))
        final = np.concatenate((mfcc, centroid, zero_crossing_rate))
        np.savetxt(fname = "data/train/features/" + str(filename[:-4]) + ".txt", X = final)
    print("Done Feature Extraction for Train")

def feature_extraction_test(hop_length):
    list_dir = sorted(os.listdir(test_dir))
    for _, filename in enumerate(list_dir):
        y, sr = librosa.load("wav/test/" + str(filename))
        mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length))
        centroid = np.array(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length))
        zero_crossing_rate = np.array(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length))
        final = np.concatenate((mfcc, centroid, zero_crossing_rate))
        np.savetxt(fname = "data/test/features/" + str(filename[:-4]) + ".txt", X = final)
    print("Done Feature Extraction for Test")

def reduce_feature_train(n_clust):
    list_dir = sorted(os.listdir("data/train/features"))
    for _,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/train/features/" + str(filename)))
        kmeans = KMeans(n_clusters = n_clust).fit(feature)
        np.savetxt(fname = "data/train/kmean_features/" + str(filename), X = np.transpose(np.array(kmeans.cluster_centers_)))
    print("Done Reducing Feature for train")

def reduce_feature_test(n_clust):
    list_dir = sorted(os.listdir("data/test/features"))
    for _,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/test/features/" + str(filename)))
        kmeans = KMeans(n_clusters = n_clust).fit(feature)
        np.savetxt(fname = "data/test/kmean_features/" + str(filename), X = np.transpose(np.array(kmeans.cluster_centers_)))
    print("Done Reducing Feature for Test")

def model_SVM(n_clust):
    list_dir = sorted(os.listdir("data/train/kmean_features"))
    output_array = []
    for i,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/train/kmean_features/" + str(filename)))
        if i==0:
            input_array = feature
        else:
            input_array = np.concatenate((input_array,feature))
        if filename[:1]=='M':
            for j in range(n_clust):
                output_array.append(1)
        elif filename[:1]=='S':
            for j in range(n_clust):
                output_array.append(0)
    output_array = np.transpose(np.array(output_array))
    clf = SVC(kernel='linear')
    clf.fit(input_array,output_array)
    joblib.dump(clf, 'finalized_svm_model.sav')

def test_SVM(n_clust):
    list_dir = sorted(os.listdir("data/test/kmean_features"))
    output_array_test = []
    for i,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/test/kmean_features/" + str(filename)))
        if i==0:
            input_array_test = feature
        else:
            input_array_test = np.concatenate((input_array_test,feature))
        if filename[:1]=='M':
            for j in range(n_clust):
                output_array_test.append(1)
        elif filename[:1]=='S':
            for j in range(n_clust):
                output_array_test.append(0)
    output_array_test = np.transpose(np.array(output_array_test))
    loaded_model = joblib.load("finalized_svm_model.sav")
    result = loaded_model.predict(input_array_test)
    conf_matrix = confusion_matrix(output_array_test, result, labels=[0,1])
    accuracy = accuracy_score(output_array_test, result)
    print("Confusion Matrix")
    print(conf_matrix)
    print("Accuracy: " + str(accuracy) + "%")



def model_MLPClassifier(n_clust):
    list_dir = sorted(os.listdir("data/train/kmean_features"))
    output_array = []
    for i,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/train/kmean_features/" + str(filename)))
        if i==0:
            input_array = feature
        else:
            input_array = np.concatenate((input_array,feature))
        if filename[:1]=='M':
            for j in range(n_clust):
                output_array.append(1)
        elif filename[:1]=='S':
            for j in range(n_clust):
                output_array.append(0)
    output_array = np.transpose(np.array(output_array))
    # clf1 = MLPClassifier(solver='lbfgs')
    clf1 = MLPClassifier(solver='lbfgs', alpha=0.00005, hidden_layer_sizes=(5, 2), random_state=1)
    # clf1 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    clf1.fit(input_array,output_array)
    joblib.dump(clf1, 'finalized_nn_model.sav')


def test_MLPClassifier(n_clust):
    list_dir = sorted(os.listdir("data/test/kmean_features"))
    output_array_test = []
    for i,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/test/kmean_features/" + str(filename)))
        if i==0:
            input_array_test = feature
        else:
            input_array_test = np.concatenate((input_array_test,feature))
        if filename[:1]=='M':
            for j in range(n_clust):
                output_array_test.append(1)
        elif filename[:1]=='S':
            for j in range(n_clust):
                output_array_test.append(0)
    output_array_test = np.transpose(np.array(output_array_test))

    loaded_model = joblib.load("finalized_nn_model.sav")
    result = loaded_model.predict(input_array_test)
    conf_matrix = confusion_matrix(output_array_test, result, labels=[0,1])
    accuracy = accuracy_score(output_array_test, result)
    print("Confusion Matrix")
    print(conf_matrix)
    print("Accuracy: " + str(accuracy) + "%")


def main():
    print("___________________________________________________________\n")
    print("Speech Music Classification")
    hop_length = 1024
    n_clust = 64
    create_dirs()
    feature_extraction_train(hop_length)
    feature_extraction_test(hop_length)
    reduce_feature_train(n_clust)
    reduce_feature_test(n_clust)
    print("SVM Modelling")
    model_SVM(n_clust)
    test_SVM(n_clust)
    print("\n")
    print("NN Modelling")
    model_MLPClassifier(n_clust)
    test_MLPClassifier(n_clust)
    print("Done Modelling!")

if __name__ == '__main__':
    main()
