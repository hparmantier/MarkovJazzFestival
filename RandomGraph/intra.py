import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import networkx as nx
import librosa
import SimulateRW as sim
import pdb
from collections import Counter

class IntraGraph():

    def __init__(self, song_file, threshold=0.7):
        self.song_file = song_file
        self.threshold = threshold
        self.y, self.sr = librosa.load(self.song_file)
        tempo, beats = librosa.beat.beat_track(self.y, self.sr, hop_length=512)
        beat_length = [beats[0]]
        for i in range(1, len(beats)):
            beat_length.append(beats[i]-beats[i-1])
        mean_beat_length = np.mean(beat_length)
        std_beat_length = np.std(beat_length)
        self.beat_duration = mean_beat_length*512/self.sr
        self.std_duration = std_beat_length*512/self.sr
        beats = np.insert(beats, 0, 0)
        self.beats = np.append(beats, int(np.ceil(len(self.y)/512)))
        self.nbeats = len(beats)
        self.fv = self.beat_features()
        self.S = self.similarity_matrix()
        self.G = self.song2graph()

    def simulate(self, save=False, filename='test.wav'):
        path = sim.generate_permutation_nx(self.G)
        y_perm = self.play_path(path)
        if save:
            librosa.output.write_wav(filename, y_perm, self.sr)

        return path

    def play_path(self, path):
        nsteps = len(path)
        y_perm = []
        for i in range(nsteps):
            
            y_perm = np.concatenate( (y_perm, self.y[ 512*self.beats[ path[i] ] : 512*self.beats[ path[i]+1 ] ]), axis=0 )

        return y_perm

    def song2graph(self):
        dt = [('weight', float)]
        sim_mat=np.matrix(self.S,dtype=dt)
        G = nx.from_numpy_matrix(sim_mat)

        new = nx.DiGraph() # directed graph

        for n in G:
            edges = list(filter(lambda e: e[0]==n, G.edges(n, data=True))) 
            #print(edges)
            edges = list(filter(lambda e: e[2]['weight'] > self.threshold,edges))
        
            if len(edges) != 0:
            
                weights = list(map(lambda e: e[2]['weight'], edges))
                maxi = max(weights)
                n_ = list(filter(lambda e: e[2]['weight']==maxi,edges))[0][1]
                new.add_edge(n,n_,weight=maxi)

            new.add_edge(n,n,weight=1.0)

        return new

    def beat_features(self):
        beat_decomposition=True

        # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        # beat_samples = librosa.frames_to_samples(beat_frames)

        ## We apply a Hamming window on every beat sample to attenuate border effects
        # if beat_decomposition:
        #     for i in range(nbeats-1):
        #         beat_length = 512*(beats[i+1]-beats[i])
        #         H = np.hamming(beat_length)
        #         y[ 512*beats[i] : 512*beats[i+1] ] = np.multiply(H, y[ 512*beats[i] : 512*beats[i+1] ] )

        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=12, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=512)

        features = np.concatenate((mfcc, chroma), axis=0)
        
        beat_features = np.zeros((24,self.nbeats))
        for i in range(self.nbeats-1):
            beat_features[:,i] = np.mean(features[:,self.beats[i]:self.beats[i+1]], axis=1)

        return beat_features

    def similarity_matrix(self, mode='affinity', width=5, filtering=True, metric='euclidean'):

        R = librosa.segment.recurrence_matrix(self.fv, width=width, metric=metric, sym=True, mode=mode)
        self.R = R
        
        R2 = R
        
        ## We filter the matrix to highlight diagonal components
        if filtering:
            for i in range(2,len(R)-2):
                for j in range(2,len(R)-2):
                    c = Counter([np.sign(R[i+t,j+t]) for t in range(-2,3)])
                    if c.most_common(1)[0][0] == 0:
                    #if ((R[i-2,j-2] == 0.0) or (R[i-1,j-1] == 0.0)) and ((R[i+1, j+1] == 0.0) or (R[i+2, j+2] == 0.0)):
                        R2[i,j] = 0.0

                           
        return R2


    def show_matrix(self, S):
        S_show = -(S-1)
        #S_show = S
        import matplotlib.pylab as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(S_show, cmap=plt.cm.gray, interpolation='None')
        plt.xlabel('Beats')
        plt.ylabel('Beats')
        #plt.colorbar()
        plt.show()


    