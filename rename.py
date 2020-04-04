import os

dir = "wav/train"

i = 1
for filename in os.listdir(dir):
    if(filename[-6:-5] == '_'):
        src = dir + "/" + str(filename)
        dst = dir + "/" + str(filename[:2]) + "0" + str(filename[2:])
        i = i + 1
        os.rename(src,dst)
