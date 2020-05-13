import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

list_dir = sorted(os.listdir("wav/train"))
file = list_dir[65]
print(file)
y = np.loadtxt(fname = "wav/train/" + file)
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
