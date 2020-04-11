import os
import librosa
import numpy as np


# To rename all the files from online
# dir = "wav/train"

# i = 1
# for filename in os.listdir(dir):
#     if(filename[-6:-5] == '_'):
#         src = dir + "/" + str(filename)
#         dst = dir + "/" + str(filename[:2]) + "0" + str(filename[2:])
#         i = i + 1
#         os.rename(src,dst)


# To extract the data files and store it in a place
# dir = "wav/test"
# for filename in os.listdir(dir):
#     y, sr = librosa.load("wav/test/" + str(filename))
#     y = np.asarray(y)
#     y = np.append(sr, y)
#     np.savetxt(fname = "workplace/wav/test/" + str(filename[:-4]) + ".txt", X = y)
#     print(filename)
