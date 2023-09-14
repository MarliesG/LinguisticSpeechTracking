import trftools
import eelbrain
import os

filename_wavfile = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/audiobook_1.wav'
# wav file can be downloaded here: https://soundcloud.com/deburen-eu/radioboek-voor-kinderen-stijn-vranken-milan?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing
fs = 64 # sampling rate by preprocessed EEG of sparrkulee dataset
directory_features = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/stimuli'

### MAKE SPECTROGRAM
stimulus_name = os.path.basename(filename_wavfile).replace('.wav', '')
path2save_spectrogram = os.path.join(directory_features, stimulus_name + '_%dHz_binned_spectrogram.pickle' % fs)

wav = eelbrain.load.wav(filename_wavfile)

# (1) make one dimensional by averaging two channels (in case of stereo audio)
if wav.has_dim('channel'):
    wav = wav.mean('channel')
wav.x = wav.x.astype(float)

# (2) lowpass filter wav file 4000 Hz
# → because of our insert phones
wav = eelbrain.filter_data(wav, low=0, high=4000)

# (3) downsample filter wav file
wav = eelbrain.resample(wav, sfreq=8000)

# (4) get gammatone spectrogram
# gammatone filterbank requires to specify the lower and higher frequency cutoff as well as the numbers of channels
# this choice is rather arbitary (usually I use 256 channels, however, this won't run on my laptop)
gammatoneSpectrogram = trftools.gammatone_bank(wav, 70, 4000, 256/2, location='left', pad=False, name='spectrogram')

# (5) bin the spectrogram
binnedSpectrogram = gammatoneSpectrogram.bin(nbins=8, dim='frequency', name='binned spectrogram')
binnedSpectrogram = binnedSpectrogram **0.6
binnedSpectrogram_resampled = eelbrain.resample(binnedSpectrogram, sfreq=fs)

eelbrain.save.pickle(binnedSpectrogram_resampled, path2save_spectrogram)

### MAKE ACOUSTIC ONSETS
path2save_acoustic_edges = os.path.join(directory_features, stimulus_name + '_%dHz_acoustic_onsets.pickle' % fs)
acoustic_edges = binnedSpectrogram.diff('time', name='acoustic onsets').clip(0)
acoustic_edges_resampled = eelbrain.resample(acoustic_edges, sfreq=fs)
eelbrain.save.pickle(acoustic_edges_resampled, path2save_acoustic_edges)