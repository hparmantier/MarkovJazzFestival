import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import networkx as nx
import librosa
import SimulateRW as sim


def affinity_matrix(song):
    y, sr = librosa.load(song)
    tempo, beats = librosa.beat.beat_track(y, sr, hop_length=512)

    nbeats = len(beats)

    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    # beat_samples = librosa.frames_to_samples(beat_frames)

    #Attenuation of borders of samples

    # for i in range(0,l):
    #     beat_index = np.mod(i,beat_length)
    #     if beat_index<border:
    #         y[i] *= beat_index/border
    #     elif beat_index>(beat_length-beat_index):
    #         y[i] *= (beat_length-beat_index)/border

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

    features = np.concatenate((mfcc, chroma), axis=0)

    beat_features = np.zeros((24,nbeats))
    beat_features[:,0] = np.mean(features[:,:beats[0]], axis=1)

    for i in range(1, nbeats-1):
        beat_features[:,i] = np.mean(features[:,beats[i]:beats[i+1]], axis=1)

    R = librosa.segment.recurrence_matrix(beat_features, mode='affinity', width=4, sym=True)
    
    R2 = R
    for i in range(2,len(R)-2):
        for j in range(2,len(R)-2):
            if ((R[i-2,j-2] == 0.0) or (R[i-1,j-1] == 0.0)) and ((R[i+1, j+1] == 0.0) or (R[i+2, j+2] == 0.0)):
                R2[i,j] = 0.0
    return R2


def build_graph(recc_matrix):
    ########################################RECURRENCE MATRIX USING LIBROSA##########################################
    # audio_file = '/home/hparmantier/Montreux Analytics/come_tog.wav'
    # y, sr = librosa.load(audio_file)
    # mfcc = librosa.feature.mfcc(y=y[0:50000], sr=sr)
    # width, height = mfcc.shape
    # recc_matrix = librosa.segment.recurrence_matrix(mfcc, mode='connectivity')
    #
    # # draw recurrence matrix
    # #librosa.display.specshow(recc_matrix[0:199, 0:199], x_axis='time', y_axis='time', aspect='equal')
    # #plt.title('Binary recurrence (connectivity)')
    # #plt.show()
    ########################################SPECTRAL CLUSTERING ON recurrence.npy####################################
    dt = [('weight', float)]
    recc_mat=np.matrix(recc_matrix,dtype=dt)
    G = nx.from_numpy_matrix(recc_matrix)
    return G
    #labels = cluster.spectral_clustering(recc_matrix, 3, eigen_solver='arpack')
    #draw_temporal_labels(labels)

def show_matrix(R):
    R_show = -(R-1)
    import matplotlib.pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(R_show, cmap=plt.cm.gray)
    plt.xlabel('Samples (beats)')
    plt.ylabel('Samples (beats)')
    plt.show()

def song2graph(song, filename):

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

def simulate(song, graph, filename):
    path = sim.generate_permutation_nx(graph)
    y, sr = librosa.load(song)
    y_perm = sim.play_path(song, path)
    librosa.output.write_wav(filename, y_perm, sr)
    return path

