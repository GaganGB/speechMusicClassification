import os
import shutil
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import joblib

# Setting the train and test directories where the dataset is stored
train_dir = "wav/train"
test_dir = "wav/test"
models_dir = "models"

def pre_modelling(hop_length = 512, n_clust = 16):
    create_dirs()
    feature_extraction_train(hop_length)
    feature_extraction_test(hop_length)
    reduce_feature_train(n_clust)
    reduce_feature_test(n_clust)

def main():
    print("Speech Music Classification")
    hop_length = 512
    n_clust = 16
    # Uncomment when running the program for first time or after changing any parameters for feature_extraction
    pre_modelling(hop_length, n_clust)
    start_SVM(hop_length, n_clust, 'SVM')
    # start_MLP(hop_length, n_clust, 'MLP')
    print("Done")

# Create new directories to store all the feature files and other related files
def create_dirs():
    if(os.path.exists("data") == True):
        try:
           shutil.rmtree("data")
        except:
           print('Error while deleting directory')
    os.mkdir("data")
    os.mkdir("data/train")
    os.mkdir("data/test")
    os.mkdir("data/train/features")
    os.mkdir("data/test/features")
    os.mkdir("data/train/kmean_features")
    os.mkdir("data/test/kmean_features")
    if(os.path.exists(models_dir) == True):
        try:
           shutil.rmtree(models_dir)
        except:
           print('Error while deleting directory')
    os.mkdir(models_dir)
    print("Done Directory creation")

"""
Extract MFCC features, spectral centroid, zero crossing rate as a vector for each utterance from train_dir directory
and store in 'data/train/features' directory
"""
def feature_extraction_train(hop_length):
    list_dir = sorted(os.listdir(train_dir))
    for _, filename in enumerate(list_dir):
        # y, sr = librosa.load("wav/train/" + str(filename))
        y = np.loadtxt(fname = "wav/train/" + str(filename))
        sr = y[0]
        y = y[1:]
        mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length))
        centroid = np.array(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length))
        zero_crossing_rate = np.array(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length))
        final = np.concatenate((mfcc, centroid, zero_crossing_rate))
        np.savetxt(fname = "data/train/features/" + str(filename[:-4]) + ".txt", X = final)
        print(filename)
    print("Done Feature Extraction for Train")

"""
Extract MFCC features, spectral centroid, zero crossing rate as a vector for each utterance from test_dir directory
and store in 'data/test/features' directory
"""
def feature_extraction_test(hop_length):
    list_dir = sorted(os.listdir(test_dir))
    for _, filename in enumerate(list_dir):
        # y, sr = librosa.load("wav/test/" + str(filename))
        y = np.loadtxt(fname = "wav/test/" + str(filename))
        sr = y[0]
        y = y[1:]
        mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length))
        centroid = np.array(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length))
        zero_crossing_rate = np.array(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length))
        final = np.concatenate((mfcc, centroid, zero_crossing_rate))
        np.savetxt(fname = "data/test/features/" + str(filename[:-4]) + ".txt", X = final)
        print(filename)
    print("Done Feature Extraction for Test")

"""
Since the extracted data has lot of feature vectors, we try to reduce the feature vectors of both train and test data using k-means
with number of cluster given as n_clust.
"""
def reduce_feature_train(n_clust):
    list_dir = sorted(os.listdir("data/train/features"))
    for _,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/train/features/" + str(filename)))
        kmeans = KMeans(n_clusters = n_clust).fit(feature)
        np.savetxt(fname = "data/train/kmean_features/" + str(filename), X = np.transpose(np.array(kmeans.cluster_centers_)))
        print(filename)
    print("Done Reducing Feature for train")

def reduce_feature_test(n_clust):
    list_dir = sorted(os.listdir("data/test/features"))
    for _,filename in enumerate(list_dir):
        feature = np.transpose(np.loadtxt(fname = "data/test/features/" + str(filename)))
        kmeans = KMeans(n_clusters = n_clust).fit(feature)
        np.savetxt(fname = "data/test/kmean_features/" + str(filename), X = np.transpose(np.array(kmeans.cluster_centers_)))
        print(filename)
    print("Done Reducing Feature for Test")

"""
We use SVM model to train and test the given dataset.
"""
def start_SVM(hop_length = 512, n_clust = 16, folder_name='SVM'):
    print("SVM Modelling")
    if(os.path.exists(models_dir + "/" + folder_name) == True):
        try:
           shutil.rmtree(models_dir + "/" + folder_name)
        except:
           print('Error while deleting directory')
    os.mkdir(models_dir + "/" + folder_name)
    model_SVM(n_clust, folder_name)
    test_SVM(n_clust, folder_name)
    print("\n")

def model_SVM(n_clust, folder_name):
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
    file_name = models_dir + "/" + folder_name + "/" + "finalized_svm_model.sav"
    joblib.dump(clf, file_name)

def test_SVM(n_clust, folder_name):
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
    file_name = models_dir + "/" + folder_name
    loaded_model = joblib.load(file_name + "/" + "finalized_svm_model.sav")
    result = loaded_model.predict(input_array_test)
    conf_matrix = confusion_matrix(output_array_test, result, labels=[0,1])
    accuracy = accuracy_score(output_array_test, result)
    f = open(file_name + "/" + "result.txt",'w')
    f.write("Confusion Matrix\n")
    f.write(str(conf_matrix[0,:]) + "\n")
    f.write(str(conf_matrix[1,:]))
    f.write("\nAccuracy: " + str(accuracy) + "%")
    f.close()

"""
We use MLP Classifier to train and test the data.
"""
def start_MLP(hop_length = 512, n_clust = 16, folder_name = 'MLP'):
    print("NN Modelling")
    if(os.path.exists(models_dir + "/" + folder_name) == True):
        try:
           shutil.rmtree(models_dir + "/" + folder_name)
        except:
           print('Error while deleting directory')
    os.mkdir(models_dir + "/" + folder_name)
    model_MLPClassifier(n_clust, folder_name)
    test_MLPClassifier(n_clust, folder_name)
    print("Done Modelling!")

def model_MLPClassifier(n_clust, folder_name):
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
    clf1 = MLPClassifier(solver='lbfgs', alpha=0.00005, hidden_layer_sizes=(5, 2), random_state=1)
    clf1.fit(input_array,output_array)
    file_name = models_dir + "/" + folder_name + "/" + 'finalized_nn_model.sav'
    joblib.dump(clf1, file_name)


def test_MLPClassifier(n_clust, folder_name):
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
    file_name = models_dir + "/" + folder_name
    loaded_model = joblib.load(file_name + "/" + "finalized_nn_model.sav")
    result = loaded_model.predict(input_array_test)
    conf_matrix = confusion_matrix(output_array_test, result, labels=[0,1])
    accuracy = accuracy_score(output_array_test, result)
    f = open(file_name + "/" + "result.txt",'w')
    f.write("Confusion Matrix\n")
    f.write(str(conf_matrix[0,:]) + "\n")
    f.write(str(conf_matrix[1,:]))
    f.write("\nAccuracy: " + str(accuracy) + "%")
    f.close()

if __name__ == '__main__':
    main()
