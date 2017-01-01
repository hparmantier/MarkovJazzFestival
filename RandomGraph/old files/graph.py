import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import networkx as nx
import librosa
import SimulateRW as sim
import pdb

def affinity_matrix(song, mode='affinity', width=5, beat_decomposition=True, filtering=True, local=False, metric='euclidean'):
    y, sr = librosa.load(song)
    tempo, beats = librosa.beat.beat_track(y, sr, hop_length=512)
    beats = np.insert(beats, 0, 0)
    beats = np.append(beats, int(np.ceil(len(y)/512)))
    nbeats = len(beats)

    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    # beat_samples = librosa.frames_to_samples(beat_frames)

    ## We apply a Hamming window on every beat sample to attenuate border effects
    # if beat_decomposition:
    #     for i in range(nbeats-1):
    #         beat_length = 512*(beats[i+1]-beats[i])
    #         H = np.hamming(beat_length)
    #         y[ 512*beats[i] : 512*beats[i+1] ] = np.multiply(H, y[ 512*beats[i] : 512*beats[i+1] ] )


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

    features = np.concatenate((mfcc, chroma), axis=0)

    if beat_decomposition:
        beat_features = np.zeros((24,nbeats))
        for i in range(nbeats-1):
            beat_features[:,i] = np.mean(features[:,beats[i]:beats[i+1]], axis=1)
        fv = beat_features # feature vectors
    else:
        fv = features


    R = librosa.segment.recurrence_matrix(fv, width=width, metric=metric, sym=True, mode=mode)
    R2 = R
    ## We filter the matrix to highlight diagonal components
    if filtering:
        for i in range(2,len(R)-2):
            for j in range(2,len(R)-2):
                if ((R[i-2,j-2] == 0.0) or (R[i-1,j-1] == 0.0)) and ((R[i+1, j+1] == 0.0) or (R[i+2, j+2] == 0.0)):
                    R2[i,j] = 0.0
                

    if local:
        for i in range(2,len(R)-2):
            for j in range(2,len(R)-2):
                if local and np.abs(i-j)>10:
                    R2[i,j] = 0.0

    return R2, beats


def build_graph(recc_matrix):
    dt = [('weight', float)]
    recc_mat=np.matrix(recc_matrix,dtype=dt)
    G = nx.from_numpy_matrix(recc_matrix)
    return G

def show_matrix(R):
    R_show = -(R-1)
    #R_show = R
    import matplotlib.pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(R_show, cmap=plt.cm.gray, interpolation='None')
    plt.xlabel('Beats')
    plt.ylabel('Beats')
    plt.colorbar()
    plt.show()

def song2graph(song, filename='test.gpickle'):

    recc_mat = affinity_matrix(song)

    G = build_graph(recc_mat)
    new = nx.DiGraph()
    threshold = 0.7

    for n in G:
        edges = list(filter(lambda e: e[0]==n, G.edges(n, data=True))) 
        print(edges)
        edges = list(filter(lambda e: e[2]['weight'] > threshold,edges))
    
        if len(edges) != 0:
        
            weights = list(map(lambda e: e[2]['weight'], edges))
            maxi = max(weights)
            n_ = list(filter(lambda e: e[2]['weight']==maxi,edges))[0][1]
            new.add_edge(n,n_,weight=maxi)

        new.add_edge(n,n,weight=1.0)

    nx.write_gpickle(new, filename)

    return new

def simulate(song, graph, filename='test.wav'):
    path = sim.generate_permutation_nx(graph)
    y, sr = librosa.load(song)
    y_perm = sim.play_path(song, path)
    librosa.output.write_wav(filename, y_perm, sr)
    return path

