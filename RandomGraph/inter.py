import os
import sys
sys.path.append(os.path.abspath('..'))
import GraphClustering.spectralClustering as sc
import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt
import intra
import librosa
import SimulateRW as sim
import scipy.stats as st
import pdb

class InterGraph():

    def __init__(self, directory='../Data/'):
        self.directory = directory
        self.song_seg = []
        self.aff = self.build_aff_matrix()
        self.G = self.songs2graph()

    def simulate(self, save=True, filename='test.wav'):
        path = sim.generate_permutation_nx(self.G)
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
            rlimit = 512*song.beats[ song.seg[self.song_seg[path[i]][1]] ]
            llimit = 512*song.beats[ song.seg[self.song_seg[path[i]][1] + 1] ]
            y_perm = np.concatenate( (y_perm, y[ rlimit : llimit ] ), axis=0 )

        return y_perm

    def build_aff_matrix(self):
        seg_list = []
        fv = [] #feature vector
        song_limits = [0]
        songs = []
        c=0
        for filename in os.listdir(self.directory):
            if filename.endswith(".mp3"):
                print(filename)
                song_graph = intra.IntraGraph(self.directory+filename)
                song_limits.append(song_limits[-1]+len(song_graph.y))
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
                c+=1
                

        featvec = np.transpose(np.asarray(fv))
        #pdb.set_trace()
        aff = librosa.segment.recurrence_matrix(featvec, width=1, metric='euclidean', sym=True, mode='affinity')
        self.songs = songs

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

    def songs2graph(self, threshold=0.6):
        dt = [('weight', float)]
        sim_mat=np.matrix(self.aff,dtype=dt)
        G = nx.from_numpy_matrix(sim_mat)

        new = nx.DiGraph() # directed graph

        for n in G:
            edges = list(filter(lambda e: e[0]==n, G.edges(n, data=True))) 
            #print(edges)
            edges = list(filter(lambda e: self.song_seg[e[0]][0] != self.song_seg[e[1]][0] and e[2]['weight'] > threshold, edges))
        
            if len(edges) != 0:
            
                weights = list(map(lambda e: e[2]['weight'], edges))
                maxi = max(weights)
                n_ = list(filter(lambda e: e[2]['weight']==maxi,edges))[0][1]
                new.add_edge(n,n_,weight=maxi)

            new.add_edge(n,n, weight=1.0)

        return new

    
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




