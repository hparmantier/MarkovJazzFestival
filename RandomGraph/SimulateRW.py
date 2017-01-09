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

def generate_permutation_nx(G, inter=False):
	n = len(G)
	path = [0]
	path = path + walker.make_n_step(G, path[0], n)
	return path

def print_path(path, file):
	f = open(file,'w')

	f.write(', '.join(map(lambda i: str(i),path))) # python will convert \n to os.linesep
	f.close() # you can omit in most cases as the destructor will


def get_format(audio):
	filename, file_extension = os.path.splitext(audio)
	return file_extension

def output_audio(audio, filename):
	file_handle = audio.export(filename, format="mp3")


def play_audio(audio):
	play(audio)
