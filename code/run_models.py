import eelbrain
import numpy as np
import os

# PARAMETERS
directory_eeg = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/eeg'
saving_directory = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/TRFs'
feature_directory = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/stimuli'

tstart = 0 # in s, from 0 ms
tstop = 0.5 # in s, to 500 ms
basis = 0.05 # in s

fs = 64 # sampling rate by preprocessed EEG of sparrkulee dataset

# get the features
spectrogram = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_binned_spectrogram.pickle'))
acoustic_onsets = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_acoustic_onsets.pickle'))
phoneme_onsets = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_phoneme_onsets.pickle'))
word_onsets = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_word_onsets.pickle'))
phoneme_surprisal = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_phoneme_surprisal_subtlex.pickle'))
cohort_entropy = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_cohort_entropy_subtlex.pickle'))
word_surprisal = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_word_surprisal_ngram.pickle'))
word_frequency = eelbrain.load.unpickle(os.path.join(feature_directory, 'audiobook_1_64Hz_word_frequency_ngram.pickle'))

models = {'baseline': [spectrogram, acoustic_onsets, phoneme_onsets, word_onsets],
          'complete': [spectrogram, acoustic_onsets, phoneme_onsets, word_onsets,
                       phoneme_surprisal, cohort_entropy, word_surprisal, word_frequency]}

# get shortest stimuli length
stimuli_length = np.min([stim.time.tmax for stim in models['complete']])

# set all stimuli to shortest stimuli length
for model in models.keys():
    models[model] = [stim.sub(time=(None, stimuli_length)) for stim in models[model]]

# CODE
files = os.listdir(directory_eeg)

for file in files:
    subject = file.split('_')[0]
    # read EEG
    eeg_data = np.load(os.path.join(directory_eeg, file))
    # make nd var from EEG
    sensor = eelbrain.Sensor.from_montage('biosemi64')[:64]
    time = eelbrain.UTS(0, 1 / fs, eeg_data.shape[1])
    eeg = eelbrain.NDVar(eeg_data.astype('float64'), (sensor, time), name=subject)

    eeg = eeg.sub(time=(None, stimuli_length))

    # run models
    for model in models.keys():
        saving_name = os.path.join(saving_directory, '%s_%s.pickle' % (subject, model))
        if os.path.exists(saving_name):
            continue
        features = models[model]
        r = eelbrain.boosting(eeg, features, tstart=tstart, tstop=tstop, basis=basis, partitions=10, test=1, selective_stopping=1)
        eelbrain.save.pickle(r, saving_name)
