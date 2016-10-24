import librosa
from librosa import display
import numpy as np
from copy import copy

n_mfcc = 20 # number of mfc coeff
chunk = 1 # number of chunks by beat
k = 30 # number of nearest-neighbors
width = 1 # suppress neighbors within +- width samples

# Find nearest neighbors in MFCC space

y, sr = librosa.load("./songs/stairway.mp3")
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=512)
print(tempo)

bps = tempo/60 
beat_length = int(np.round(sr/bps/chunk)) # number of samples in a beat

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

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=beat_length)
R = librosa.segment.recurrence_matrix(mfcc, sym=True)

# Suppress neighbors within +- 7 samples

#R = librosa.segment.recurrence_matrix(mfcc, width=7)

# Use cosine similarity instead of Euclidean distance

#R = librosa.segment.recurrence_matrix(mfcc, metric='cosine')

# Require mutual nearest neighbors

#R = librosa.segment.recurrence_matrix(mfcc, sym=True)


#Use an affinity matrix instead of binary connectivity

# R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity')

# Plot the feature and recurrence matrices

[x,y] = R.shape
R2 = copy(R)

w = 3 # min sequence size

for i in range(0,x):
	for j in range(0,y):
		if np.abs(i-j)==1:
			R2[i,j] = 1
		elif i<=w or j<=w or i>=x-w or j>=y-w:
			R2[i,j] = 0
		else:
			c = 0 #counter
			for t in range(-w,w+1):
				c += R[i+t, j+t]
			R2[i,j] = int(np.round(c/(2*w+1)))

import matplotlib.pylab as plt
R = -1*(R-1)
R2 = -1*(R2-1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(R2, cmap=plt.cm.gray)
plt.xlabel('Samples (beats)')
plt.ylabel('Samples (beats)')
plt.show()

np.save('mfcc_stairway.npy', mfcc)
np.save('R_stairway.npy', R)
