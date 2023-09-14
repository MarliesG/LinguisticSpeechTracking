import os
import pandas as pd
import codecs
import numpy as np

## PARAMETERS
subtlex_file = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/SUBTLEX-NL.cd-above2.txt'
saving_name = '/home/luna.kuleuven.be/u0123908/Documents/Seminars/2023_09_CNSP_workshop/data/other/'

n_words_per_part = 100000 - 100 # maximum amount of words accepted per file by Web MAUS (if using the maximum amount, the web tool sometimes crashes)

## CODE
df = pd.read_csv(subtlex_file, sep="\t", encoding='ISO-8859-1')

# select only entries which occur more than 200 times in the whole corpus (this is an arbitary set value to remove typos and weird words)
df = df[df.FREQcount > 200]
df = df[[not ('&' in word) for word in df.Word]]

words = df.Word.to_list()
n_parts = int(np.ceil(len(words) / n_words_per_part))

i_word = 0
for i_part in range(0, n_parts):

    with codecs.open(os.path.join(saving_name, 'subtlexWORDS_%d.txt' % (i_part+1)), 'w', encoding='UTF-8') as fid:
        i = 0
        while i < n_words_per_part and i_word < len(words):

            word = words[i_word]

            fid.write('%s\n' % word)
            i_word += 1
            i += 1

# the output file is used to generate the pronunciation dictionary using the BAS web services, more specifically:
# the G2P service: https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/Grapheme2Phoneme
# in this case specifying the language to Dutch (NL) and using the option lex → this results in a file with all the words
# and the corresponding phoneme segmentation