import eelbrain
import textgrids
import numpy as np
import os

textgrid_file = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/audiobook_1.TextGrid'
fs = 64 # sampling rate by preprocessed EEG of sparrkulee dataset
directory_features = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/stimuli'

stimulus_name = os.path.basename(textgrid_file).replace('.TextGrid', '')

### MAKE WORD ONSETS
# load the txt file
grid = textgrids.TextGrid(textgrid_file)
word_onset_times = [interval.xmin for interval in grid['ORT-MAU'] if interval.text not in ['']]

time = eelbrain.UTS.from_int(0,  np.ceil(grid.xmax*fs), fs)
stimulus_wordOnsets = eelbrain.NDVar(np.zeros(len(time)), time, name='word onsets')
stimulus_wordOnsets[word_onset_times] = 1

path2save_word_onsets = os.path.join(directory_features, stimulus_name + '_%dHz_word_onsets.pickle' % fs)
eelbrain.save.pickle(stimulus_wordOnsets, path2save_word_onsets)

### MAKE PHONEME ONSETS
phonemes_onset_times = [interval.xmin for interval in grid['MAU'] if
                        interval.text not in ['<p:>', '<p>', '<nib>', '<usb>']]

stimulus_phonemeOnsets = eelbrain.NDVar(np.zeros(len(time)), time, name='phoneme onsets')
stimulus_phonemeOnsets[phonemes_onset_times] = 1

path2save_phoneme_onsets = os.path.join(directory_features, stimulus_name + '_%dHz_phoneme_onsets.pickle' % fs)
eelbrain.save.pickle(stimulus_phonemeOnsets, path2save_phoneme_onsets)