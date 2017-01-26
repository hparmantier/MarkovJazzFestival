import os
import sys
from numpy.random import choice
sys.path.append(os.path.abspath('..'))
#import GraphClustering.spectralClustering as sc
import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt
import intra
import librosa
import SimulateRW as sim
import scipy.stats as st
import pdb
import pickle

class InterGraph():

    def __init__(self, save=False, directory='../../../RS50/', threshold=0.5, max_songs=100):
        self.directory = directory
        self.threshold = threshold
        self.max_songs = max_songs
        self.song_seg = []
        self.aff = self.build_aff_matrix()
        self.G = self.songs2graph()
        self.im = self.infinite_medley()
        self.beat_duration = self.songs[0].beat_duration
        if save:
            with open('rs.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def simulate_medley(self, first_song=0, save=False):
        path = sim.generate_permutation_nx(self.im, inter=False)
        y_perm = self.play_medley(path)
        print(y_perm)
        if save:
            print('save')
            librosa.output.write_wav('test.wav', y_perm, self.songs[0].sr)

        return path

    def simulate_inter(self, save=False, filename='test.wav'):
        path = sim.generate_permutation_nx(self.G, inter=True)
        y_perm = self.play_path(path)
        if save:
            print('save')
            librosa.output.write_wav(filename, y_perm, self.songs[0].sr)

        return path

    def play_path(self, path):
        nsteps = len(path)
        y_perm = []
        for i in range(nsteps):
            song = self.songs[ self.song_seg[path[i]][0] ]
            y = song.y
            print(np.mean(y))
            rlimit = 512*song.beats[ song.seg[self.song_seg[path[i]][1]] ]
            llimit = 512*song.beats[ song.seg[self.song_seg[path[i]][1] + 1] ]
            y_perm = np.concatenate( (y_perm, y[ rlimit : llimit ] ), axis=0 )

        return y_perm

    def play_medley(self, path):
        nsteps = len(path)
        y_perm = []
        for i in range(nsteps):
            song_index=0
            while path[i]>=self.song_limits[song_index+1]:
                song_index += 1
            song = self.songs[song_index]
            beat_index = path[i] - self.song_limits[song_index]
            y = song.y
            rlimit = 512*song.beats[beat_index]
            llimit = 512*song.beats[beat_index+1]
            y_perm = np.concatenate( (y_perm, y[rlimit : llimit]), axis=0)

        return y_perm


    def build_aff_matrix(self):
        seg_list = []
        fv = [] #feature vector
        song_limits = [0]
        songs = []
        all_beats = []
        c=0
        shift = 0
        for filename in os.listdir(self.directory):
            if filename.endswith(".mp3"):
                print(filename)
                song_graph = intra.IntraGraph(self.directory+filename)
                song_limits.append(song_limits[-1]+song_graph.nbeats)
                #print(song_limits)
                features = song_graph.beat_features()
                seg = self.segment_song(song_graph.similarity_matrix(), song_graph.nbeats)
                for i in range(len(seg)-1):
                    print(i)
                    mean_feat = np.mean(features[:, seg[i] : seg[i+1]], axis=1)
                    fv.append(mean_feat)
                    self.song_seg.append((c,i))
                song_graph.seg = seg
                songs.append(song_graph)
                all_beats.append(seg+shift*np.ones(len(seg)))
                shift += song_graph.nbeats
                c+=1
                if c>=self.max_songs:
                    break


        featvec = np.transpose(np.asarray(fv))
        #pdb.set_trace()
        aff = librosa.segment.recurrence_matrix(featvec, width=1, metric='euclidean', sym=True, mode='affinity')
        self.songs = songs
        self.nsongs = len(songs)
        self.song_limits = song_limits
        self.all_beats = [int(item) for sublist in all_beats for item in sublist]
        print(self.all_beats)

        return aff

    def gkern(self, kernlen=21, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel

    def segment_song(self, R, nbeats, rs=14, threshold=0.6, viz=False):
        """Returns a list of beats that are border between two homogeneous song parts"""

        [r, c] = R.shape
        gk = self.gkern(2*rs+1, nsig=1)
        gk = gk/np.max(gk)
        mgk = -gk
        c1 = gk[:rs,:rs]
        c2 = mgk[:rs,rs+1:]
        c3 = mgk[rs+1:,:rs]
        c4 = gk[rs+1:,rs+1:]

        N = np.zeros((r,))
        neighborhood = np.zeros((rs,rs))
        for i in range(rs+1, r-rs-1):
            neighborhood = R[i-rs:i+rs,i-rs:i+rs]
            neigh_sum = np.sum(neighborhood)
            for m in range(rs):
              for n in range(rs):
                 N[i] += c1[m][n]*R[i-rs+m][i-rs+n]+c2[m][n]*R[i+m+1][i+n-rs]+c3[m][n]*R[i+m-rs][i+n+1]+c4[m][n]*R[i+m+1][i+n+1]
            N[i] /= neigh_sum+0.0001

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

        return seg_beats

    def test_segmentation(self, song):
        song_graph = intra.IntraGraph(song)
        S = song_graph.similarity_matrix(filtering=False)
        seg_beats = self.segment_song(S, song_graph.nbeats)
        print(seg_beats)
        for i in range(len(seg_beats)-1):
            filename = 'wall_%d.wav' % i
            librosa.output.write_wav(filename, song_graph.y[512*song_graph.beats[seg_beats[i]] : 512*song_graph.beats[seg_beats[i+1]]], song_graph.sr)

    def songs2graph(self):
        dt = [('weight', float)]
        sim_mat=np.matrix(self.aff,dtype=dt)
        G = nx.from_numpy_matrix(sim_mat)

        new = nx.DiGraph() # directed graph

        for n in G:
            edges = list(filter(lambda e: e[0]==n, G.edges(n, data=True)))
            #print(edges)
            edges = list(filter(lambda e: self.song_seg[e[0]][0] != self.song_seg[e[1]][0] and e[2]['weight'] > self.threshold, edges))

            if len(edges) != 0:

                weights = list(map(lambda e: e[2]['weight'], edges))
                maxi = max(weights)
                n_ = list(filter(lambda e: e[2]['weight']==maxi,edges))[0][1]
                new.add_edge(n,n_,weight=maxi)

            new.add_edge(n,n, weight=1.0)

        return new

    def infinite_medley(self):
        G = nx.DiGraph()
        shift = 0
        for song in self.songs:
            for e in song.G.edges(data=True):
                G.add_node(e[0]+shift, music=song.song_file)
                G.add_node(e[1]+shift, music=song.song_file)
                G.add_edge(e[0]+shift, e[1]+shift, weight=e[2]['weight'])

            shift += len(song.G)
        shift = 0
        for e in self.G.edges(data=True):
            e1 = self.all_beats[int(e[0]+1)]
            e2 = self.all_beats[int(e[1])]
            G.add_edge(e1, e2, weight=e[2]['weight'])

        return G

    def show_matrix(self, S):
        S_show = -(S-1)
        #S_show = S
        import matplotlib.pylab as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(S_show, cmap=plt.cm.gray, interpolation='None')
        plt.xlabel('Segments')
        plt.ylabel('Segments')
        #plt.colorbar()
        plt.show()
