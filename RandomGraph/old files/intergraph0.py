import os
import sys
sys.path.append(os.path.abspath('..'))
import GraphClustering.spectralClustering as sc
import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt
import graph
import librosa
import scipy.stats as st
#import pdb

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def segment_song(R, nbeats, rs=14, threshold=0.6, viz=False):
    """Returns a list of beats that are border between two homogeneous song parts"""

    
    [r, c] = R.shape
    gk = gkern(2*rs+1, nsig=2)
    gk = gk/np.max(gk)
    mgk = -gk
    c1 = gk[:rs,:rs]
    c2 = mgk[:rs,rs+1:]
    c3 = mgk[rs+1:,:rs]
    c4 = gk[rs+1:,rs+1:]

    C1 = np.concatenate((c1, c3), axis=0)
    C2 = np.concatenate((c2, c4), axis=0)
    C = np.concatenate((C1, C2), axis=1)

    N = np.zeros((r,))
    neighborhood = np.zeros((rs,rs))
    for i in range(rs+1, r-rs-1):
        neighborhood = R[i-rs:i+rs,i-rs:i+rs] 
        neigh_sum = np.sum(neighborhood) 
        for m in range(rs):
          for n in range(rs):
             N[i] += c1[m][n]*R[i-rs+m][i-rs+n]+c2[m][n]*R[i+m+1][i+n-rs]+c3[m][n]*R[i+m-rs][i+n+1]+c4[m][n]*R[i+m+1][i+n+1] 
        N[i] /= neigh_sum

    seg_beats = []

    for i in range(rs+1, r-rs-1):
        if N[i]>threshold and N[i]==np.max(N[i-rs:i+rs]):
            seg_beats = np.append(seg_beats, i)

    seg_beats = np.insert(seg_beats, 0, 0)
    seg_beats = np.append(seg_beats, nbeats-1)

    if viz:
        import matplotlib.pylab as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #ax.set_aspect('equal')
        plt.xlabel('Beats')
        plt.ylabel('Novelty score')
        plt.plot(N)
        plt.show()

    return seg_beats, C

def test_segmentation(song):
    y, sr = librosa.load(song)
    R, beats = graph.affinity_matrix(song=song, mode='affinity', metric='euclidean', width=1, filtering=False, local=False)
    seg_beats = segment_song(R, len(beats))
    print(seg_beats)
    for i in range(len(seg_beats)-1):
        filename = 'wall_%d.wav' % i 
        librosa.output.write_wav(filename, y[512*beats[seg_beats[i]] : 512*beats[seg_beats[i+1]]], sr)

    
def build_aff_matrix(directory='../Data/'):
    seg_list = []
    fv = [] #feature vector
    song_limits = [0]
    c=0
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            print(filename)
            song = directory+filename
            y, sr = librosa.load(song)
            song_limits.append(song_limits[-1]+len(y))
            print(song_limits)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=512)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            features = np.concatenate((mfcc, chroma), axis=0)
            R, beats = graph.affinity_matrix(song=song, mode='affinity', metric='euclidean', width=1, filtering=False, local=False)
            seg = segment_song(R, len(beats))
            for i in range(len(seg)-1):
                print(i)
                fv.append(np.mean(features[:, beats[seg[i]] : beats[seg[i+1]]], axis=1).tolist())

    featvec = np.transpose(np.asarray(fv))
    print(featvec.shape)
    aff = librosa.segment.recurrence_matrix(featvec, width=1, metric='euclidean', sym=True, mode='affinity')

    return aff

def generate_track():
    


