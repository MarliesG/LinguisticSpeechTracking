import eelbrain
import pandas as pd
import os
import numpy as np

## PARAMETERS
textgrid_file = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/audiobook_1.TextGrid'
directory_features = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/stimuli'
fs = 64 # sampling rate by preprocessed EEG of sparrkulee dataset

ngram_outcome_file = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/ngram_audiobook_1.txt'

## CODE
def make_ngram_variables(file, fs):
    # load the txt file -- change header
    header_list = ["word", "tstart", "tstop","frequency","surprisal"]
    df = pd.read_csv(file, sep='\t', names=header_list, usecols=[0, 1, 2, 3, 4], header=0)

    # initialize the ND vars
    time = eelbrain.UTS.from_int(0,  np.ceil((df.values[len(df)-1, 2]+2)*fs), fs) # take 2s extra
    stimulus_wordFrequency = eelbrain.NDVar(np.zeros(len(time)), time, name='word_frequency_ngram')
    stimulus_surprisal = eelbrain.NDVar(np.zeros(len(time)), time, name='word_surprisal_ngram')

    # find indices of word onsets
    startTimes = df['tstart']
    allWFValues = df['frequency']
    allSurprisalValues = df['surprisal']


    for start, WF, SV in zip(startTimes, allWFValues, allSurprisalValues):
        stimulus_wordFrequency[start] = WF
        stimulus_surprisal[start] = SV

    return stimulus_wordFrequency, stimulus_surprisal

if __name__ == '__main__':
    stimulus_wf, stimulus_surprisal = make_ngram_variables(ngram_outcome_file, fs)

    basename = os.path.basename(ngram_outcome_file).replace('ngram_', '')
    eelbrain.save.pickle(stimulus_surprisal,
                             os.path.join(directory_features, basename.replace('.txt','_%dHz_word_surprisal_ngram.pickle' % fs)))
    eelbrain.save.pickle(stimulus_wf,
                             os.path.join(directory_features, basename.replace('.txt','_%dHz_word_frequency_ngram.pickle' % fs)))




