import os
import sys
#helper = '/home/hparmantier/Montreux Analytics/Scripts'
#sys.path.append(os.path.abspath(helper))
#import Neo4jInterface.read_db as read
import numpy as np
import RandomStep as walker
#from pydub import AudioSegment
#from pydub.playback import play
import math
import os
import librosa



#temporary design: walker run for |beats| number of times
def generate_permutation(music):
	G = read.stream_built_nx(music)
	n = len(G)
	path = [0]
	path = path + walker.make_n_step(G, path[0], n)
	return path

def generate_permutation_nx(G):
	n = len(G)
	path = [0]
	path = path + walker.make_n_step(G, path[0], n)
	return path

def print_path(path, file):
	f = open(file,'w')

	f.write(', '.join(map(lambda i: str(i),path))) # python will convert \n to os.linesep
	f.close() # you can omit in most cases as the destructor will


def play_permutation(audio, permut, beat_nb):
	extension = get_format(audio)
	song = AudioSegment.from_file(audio, format=extension)
	samples = song.get_array_of_samples()
	beat_size = math.ceil(len(samples) / beat_nb)
	#beats = zip(*[iter(song)]*beat_size)
	beats = [song[i:i+beat_size] for i in range(0,len(samples),beat_size)]
	song_permuted = [beats[permut[i]] for i in range(len(permut))]
	#return song_permuted
	#return [val for beat in song_permuted for val in beat]

def play_permutation_bis(audio, permut, beat_nb):
	extension = get_format(audio)
	song = AudioSegment.from_file(audio, format=extension)
	samples = song.get_array_of_samples()
	beat_size = math.ceil(len(samples) / beat_nb)
	#beats = zip(*[iter(song)]*beat_size)
	beats = [samples[i:i+beat_size] for i in range(0,len(samples),beat_size)]
	song_permuted = [beats[permut[i]] for i in range(len(permut))]
	#new_samples = song_permuted.tostring()
	return song._spawn(np.asarray(song_permuted).tostring())

def play_path(song, path):
	y, sr = librosa.load(song)
	oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
	tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=512)
	bps = tempo/60
	beat_length = int(np.round(sr/bps))
	y_perm = y
	for i in range(0, int(np.floor(len(y)/beat_length))):
		y_perm[i*beat_length:(i+1)*beat_length] = y[path[i]*beat_length:(path[i]+1)*beat_length]

	return y_perm


def get_format(audio):
	filename, file_extension = os.path.splitext(audio)
	return file_extension

def output_audio(audio, filename):
	file_handle = audio.export(filename, format="mp3")


def play_audio(audio):
	play(audio)
