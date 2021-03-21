from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf

DATADIR = '/Users/gt/Documents/GitHub/asr/decoding/165_natural_sounds/165_natural_sounds/'

files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]

# Resample 165 sounds to 16kHz
for file in wav_files:
	data = wavfile.read(DATADIR + file)
	framerate = data[0]
	sounddata = data[1]
	time = np.arange(0, len(sounddata)) / framerate
	print('Sample rate:', framerate, 'Hz')
	print('Total time:', len(sounddata) / framerate, 's')
	
	# Resample to 16 kHz
	audio_input, _ = librosa.load(DATADIR + file, sr=16000)
	sf.write(f'/Users/gt/Documents/GitHub/control-neural/data/stimuli/165_natural_sounds_16kHz/{file}', audio_input,
			 16000)