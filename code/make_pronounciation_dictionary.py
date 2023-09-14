import pandas as pd
import glob
import os
import numpy as np
import eelbrain

# http://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSGetInventar?LANGUAGE=deu

# PARAMETERS
directory_segmentation = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other'

# CODE
# get the .csv files
par_files = glob.glob(os.path.join(directory_segmentation, '*.csv'))

# find longest phoneme sequence
df_to_concat = []
for par_file in par_files:
    df = pd.read_csv(par_file, sep=";", header=None, names=['WORD', 'PHONEME', 'N1', 'N2', 'N3'])

    words = df.WORD.to_list()
    words = [word.lower() for word in words]
    phoneme_segmentations = [phoneme.split(' ') for phoneme in df.PHONEME.to_list()]

    assert len(words) == len(phoneme_segmentations)

    longest_phoneme_segmentation = np.max([len(phoneme_segmentation) for phoneme_segmentation in phoneme_segmentations])
    d = {'WORD': words}
    for phoneme_idx in range(1, longest_phoneme_segmentation + 1):
        # make entry in dict
        d['P%d' % phoneme_idx] = []
        for phoneme_segmentation in phoneme_segmentations:
            if len(phoneme_segmentation) >= phoneme_idx:
                # python start counting from 0
                d['P%d' % phoneme_idx].append(phoneme_segmentation[phoneme_idx -1])
            else:
                d['P%d' % phoneme_idx].append(None)

    df_phoneme_word = pd.DataFrame(data=d)
    df_to_concat.append(df_phoneme_word)

# combine both df
# set both df to the same size
longest_shape = np.max([ds.shape[1] for ds in df_to_concat])
for ds in df_to_concat:
    if ds.shape[1] < longest_shape:
        # append None columns
        difference_in_columns = longest_shape - ds.shape[1]
        for ind in range(difference_in_columns):
            phoneme_idx = longest_shape - 1 - (difference_in_columns-1 - ind)
            ds['P%d' % phoneme_idx] = None

df_concat = pd.concat(df_to_concat, axis=0)

# save
eelbrain.save.pickle(df_concat, '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/phonemeDictionary-NL.pickle')