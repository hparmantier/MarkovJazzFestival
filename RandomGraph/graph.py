import numpy as np
import networkx as nx
import librosa


def affinity_matrix(song):
    y, sr = librosa.load(song)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=512)

    bps = tempo/60 
    beat_length = int(np.round(sr/bps)) # number of samples in a beat

    l = y.shape[0]
    border = np.ceil(0.1*beat_length) # we will use it to attenuate borders of each sample

    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    # beat_samples = librosa.frames_to_samples(beat_frames)

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

    feature = np.concatenate((mfcc, chroma), axis=0)

    R = librosa.segment.recurrence_matrix(feature, mode='affinity', width=5, sym=True)
    c = 0
    argmaxs = [R[i].argmax() for i in range(0, len(R))]
    maxs = [R[i][argmaxs[i]] for i in range(0, len(R))]

    R2 = np.zeros((len(R), len(R)))

    for i in range(0, len(R)):
        j = argmaxs[i]
        m = maxs[i]
        if(m>0.5):
            R2[i][i] = 1.5-m
            R2[i][j] = m-0.5

        else:
            R2[i][i] = 1 

    R2[-1][0] = 1

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