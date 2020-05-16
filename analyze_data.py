import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

starttime = time.time()

def single_spectrogram():
    list_dir = sorted(os.listdir("../data/normalized_data/train"))
    file = list_dir
    print(file)
    y = np.loadtxt(fname = "../data/normalized_data/train/" + file)
    sr = y[0]
    y = y[1:]
    chroma = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, window='hann')
    chroma_dB = librosa.power_to_db(chroma, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma_dB, y_axis='mel', x_axis='time', sr=sr, fmax = 8000)
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()

def check_feature(filename):
    result = []
    f = open("onsets.txt", 'a')
    y = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
    result=np.shape(librosa.onset.onset_detect(y, sr=22050))
    print(filename)
    f.write(filename + ": " + str(result) + "\n")
    f.close()
    return result

def beat_track(filename):
    result = []
    f = open("beat_track.txt", 'a')
    y = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
    result=librosa.beat.beat_track(y, sr=22050)
    print(filename)
    f.write(filename + ": " + str(result) + "\n")
    f.close()
    return result

def rms(filename):
    result = []
    f = open("zcr.txt", 'a')
    y = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
    result=librosa.feature.zero_crossing_rate(y)[0][20:]
    beatOfRms=np.std(result)
    # print(np.shape(result))
    # t = np.arange(0, np.shape(result)[0])
    # print(filename)
    # f.write(filename + ": " + str(result) + "\n")
    # f.close()
    # print(result)
    # print(t)
    # plt.figure()
    # plt.plot(t, result)
    # plt.savefig("zcr/" + filename[:-4] + ".png")
    # plt.close()
    return beatOfRms

def spectrogram(filename):
    result = []
    y = np.loadtxt(fname = "../data/normalized_data/train/" + str(filename))
    result=librosa.feature.melspectrogram(y=y, sr=22050, fmax=6000)
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(result, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, fmax=6000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig("spec/"+filename[:-4]+".png")
    plt.close()
    print(filename)
    return 0

def do_task():
    pool = mp.Pool(mp.cpu_count())
    list_dir = sorted(os.listdir("../data/normalized_data/train"))
    results = pool.map(spectrogram, [name for name in list_dir])

    pool.close()

    # label = np.concatenate((np.zeros(52), np.zeros(52)+1))
    # y = np.concatenate((np.zeros(52), np.zeros(52)+10))
    # plt.scatter(x=results, y=y, label=label)
    # plt.title("STD of ZCR")
    # plt.show()

do_task()

print('That took {} seconds'.format(time.time() - starttime))
# check_feature()

