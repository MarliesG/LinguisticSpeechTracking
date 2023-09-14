import eelbrain
import os
import textgrids
import numpy as np
import util_phoneme_linguistics

## PARAMETERS
textgrid_file = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/audiobook_1.TextGrid'
directory_features = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/stimuli'
fs = 64 # sampling rate by preprocessed EEG of sparrkulee dataset

subtlex_filename = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/SUBTLEX-NL.cd-above2.txt'
pronounciation_dict_filename = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/phonemeDictionary-NL.pickle'

## CODE

def flatten(l):
    return [item for sublist in l for item in sublist]

grid = textgrids.TextGrid(textgrid_file)

# get words and phoneme segmentation
all_phoneme_onsets = []
all_phoneme_surprisal = []
all_cohort_entropy = []
for word_info in grid['ORT-MAU']:
    word = word_info.text
    word_start = word_info.xmin
    word_end = word_info.xmax

    phoneme_segmentation = [phoneme_info.text for phoneme_info in grid['MAU'] if (phoneme_info.xmin >= word_start and
                                                                             phoneme_info.xmax <= word_end and
                                                                             phoneme_info.text not in ['<p:>', '<p>', '<nib>', '<usb>'])]
    phoneme_onset_times = [phoneme_info.xmin for phoneme_info in grid['MAU'] if (phoneme_info.xmin >= word_start and
                                                                             phoneme_info.xmax <= word_end and
                                                                             phoneme_info.text not in ['<p:>', '<p>', '<nib>', '<usb>'])]

    if not phoneme_segmentation:
        # skip empty segmentation (for example silences)
        continue

    phoneme_surprisal, cohort_entropy = util_phoneme_linguistics.calculate_phoneme_linguistics(word, phoneme_segmentation, subtlex_filename, pronounciation_dict_filename)

    # when the word is not present in the pronounciation dict → inf or nan value
    phoneme_surprisal = [0 if np.isnan(elem) else elem for elem in phoneme_surprisal]
    cohort_entropy = [0 if np.isnan(elem) else elem for elem in cohort_entropy]

    phoneme_surprisal = [0 if np.isinf(elem) else elem for elem in phoneme_surprisal]
    cohort_entropy = [0 if np.isinf(elem) else elem for elem in cohort_entropy]

    assert len(phoneme_onset_times) == len(phoneme_surprisal)
    assert len(phoneme_onset_times) == len(cohort_entropy)

    all_phoneme_onsets.append(phoneme_onset_times)
    all_phoneme_surprisal.append(phoneme_surprisal)
    all_cohort_entropy.append(cohort_entropy)

# flatten the arrays
all_phoneme_onsets = flatten(all_phoneme_onsets)
all_phoneme_surprisal = flatten(all_phoneme_surprisal)
all_cohort_entropy = flatten(all_cohort_entropy)

# make directory for features
time = eelbrain.UTS.from_int(0,  np.ceil(grid.xmax*fs), fs)
stimulus_phoneme_surprisal = eelbrain.NDVar(np.zeros(len(time)), time, name='phoneme_surprisal_subtlex')
stimulus_cohort_entropy = eelbrain.NDVar(np.zeros(len(time)), time, name='cohort_entropy_subtlex')

stimulus_phoneme_surprisal[all_phoneme_onsets] = all_phoneme_surprisal
stimulus_cohort_entropy[all_phoneme_onsets] = all_cohort_entropy

# save nd var
eelbrain.save.pickle(stimulus_phoneme_surprisal, os.path.join(directory_features, os.path.basename(textgrid_file).replace('.TextGrid',
                                                                                                   '_%dHz_phoneme_surprisal_subtlex.pickle' % fs)))
eelbrain.save.pickle(stimulus_cohort_entropy,
                     os.path.join(directory_features, os.path.basename(textgrid_file).replace('.TextGrid',
                                                                                      '_%dHz_cohort_entropy_subtlex.pickle' % fs)))