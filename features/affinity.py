import librosa
import numpy as np
from copy import copy

y, sr = librosa.load("./songs/creep.mp3")
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=512)
print(tempo)

bps = tempo/60 
beat_length = int(np.round(sr/bps)) # number of samples in a beat

l = y.shape[0]
border = np.ceil(0.1*beat_length) # we will use it to attenuate borders of each sample

#Attenuation of borders of samples

for i in range(0,l):
	beat_index = np.mod(i,beat_length)
	if beat_index<border:
		y[i] *= beat_index/border
	elif beat_index>(beat_length-beat_index):
		y[i] *= (beat_length-beat_index)/border

n_sample = beat_length*np.ceil(y.size/beat_length) # number of samples of y we keep

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=beat_length)
chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=beat_length)
print(mfcc.shape)
print(chroma.shape)
feature = np.concatenate((mfcc, chroma), axis=0)
print(feature.shape)

R = librosa.segment.recurrence_matrix(feature, mode='affinity', width=5, sym=True)

import matplotlib.pylab as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(R, cmap=plt.cm.gray)
plt.xlabel('Samples (beats)')
plt.ylabel('Samples (beats)')
plt.show()