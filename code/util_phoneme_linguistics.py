import pandas as pd
import eelbrain
import numpy as np

def calculate_phoneme_linguistics(word, phoneme_segmentation, subtlex_filename, pronounciation_dict_filename):
    word = word.lower()

    ## load subtlex + set all words to lower
    df_subtlex = pd.read_csv(subtlex_filename, sep="\t", encoding='ISO-8859-1')
    # select only entries with freqcount > 1
    df_subtlex = df_subtlex[df_subtlex.FREQcount > 200]
    df_subtlex = df_subtlex[[not ('&' in word) for word in df_subtlex.Word]]
    df_subtlex.Word = [word.lower() for word in df_subtlex.Word.to_list()]

    ## load pronounciation_dict
    df_pronounciation = eelbrain.load.unpickle(pronounciation_dict_filename)
    df_pronounciation.WORD = [word.lower() for word in df_pronounciation.WORD.to_list()]

    # the pronounciation dict is smaller (check out with [word for word in df_subtlex.Word if word not in df_pronounciation.WORD.values]
    # you will see that these word do not make sense, contain weird symbols or underscores)
    df_subtlex = df_subtlex[[word in df_pronounciation.WORD.values for word in df_subtlex.Word.to_list()]]

    ## calculate phoneme linguistic features
    phoneme_surprisal_all_phonemes = []
    cohort_entropy_all_phonemes = []
    for phoneme_idx in range(1, len(phoneme_segmentation) + 1):
        # PHONEME DICTIONARY STARTS AT P1 (NOT P0) → index is too high
        phoneme_surprisal = calculate_phoneme_surprisal(phoneme_idx, phoneme_segmentation, df_pronounciation, df_subtlex)
        cohort_entropy = calculate_cohort_entropy(phoneme_idx, phoneme_segmentation, df_pronounciation, df_subtlex)

        phoneme_surprisal_all_phonemes.append(phoneme_surprisal)
        cohort_entropy_all_phonemes.append(cohort_entropy)

    return phoneme_surprisal_all_phonemes, cohort_entropy_all_phonemes


def calculate_phoneme_surprisal(phoneme_idx, phonemes_of_word, df_pronounciation, df_subtlex):
    # index to python! start at 0!!
    current_phoneme = phonemes_of_word[phoneme_idx - 1]
    if phoneme_idx == 1:
        # determine frequency at the beginning
        words_active_cohort = df_pronounciation.WORD[df_pronounciation.P1 == current_phoneme]
        # get sum of occurences of the words
        word_count_active_cohort = get_occurence_count(words_active_cohort, df_subtlex)
        total_word_count = sum(df_subtlex.FREQcount)
        phoneme_surprisal = -np.log10(word_count_active_cohort / total_word_count)
    else:
        # get the active cohort of the PREVIOUS PHONEME
        word_active_cohort_previous_phoneme = get_active_cohort(phoneme_idx - 1, phonemes_of_word, df_pronounciation)
        word_count_previous_cohort = get_occurence_count(word_active_cohort_previous_phoneme, df_subtlex)

        words_active_cohort_current_phoneme = get_active_cohort(phoneme_idx, phonemes_of_word, df_pronounciation)
        word_count_current_cohort = get_occurence_count(words_active_cohort_current_phoneme, df_subtlex)

        assert word_count_previous_cohort >= word_count_current_cohort

        if not (word_count_previous_cohort==0 or word_count_current_cohort==0):
            phoneme_surprisal = -np.log10(word_count_current_cohort / word_count_previous_cohort)
        else:
            phoneme_surprisal = 0
    return phoneme_surprisal

def get_active_cohort(phoneme_idx, phonemes_of_word, df_pronounciation):
    included_words = np.ones(df_pronounciation.shape[0], dtype=bool)
    for p_idx in range(1, phoneme_idx + 1):
        # find included words for this phoneme
        current_phoneme = phonemes_of_word[p_idx - 1]
        ph_cohort = df_pronounciation['P%d' % (p_idx)] == current_phoneme
        included_words = np.logical_and(included_words, ph_cohort)

    word_cohort = df_pronounciation.WORD[included_words]
    return word_cohort


def calculate_cohort_entropy(phoneme_idx, phoneme_segmentation, df_pronounciation, df_subtlex):
    words_active_cohort = get_active_cohort(phoneme_idx, phoneme_segmentation, df_pronounciation)
    word_count_cohort = get_occurence_count(words_active_cohort, df_subtlex)

    cohort_entropy = 0
    check = 0
    for word in words_active_cohort.values:
        word_count = get_occurence_count([word], df_subtlex)
        if word_count == 0:
            continue
        word_probability = word_count / word_count_cohort
        cohort_entropy -= word_probability*np.log10(word_probability)
        check += word_count

    assert word_count_cohort == check

    return cohort_entropy

def get_occurence_count(words, df_subtlex):
    occurences = 0
    for word in words:
        if not df_subtlex.FREQcount[df_subtlex.Word == word].shape[0] == 1:
            # some words are transformed due to the MAUS application and happen to be misspelled or introduce a weird
            # character as ã
            continue
        word_count = df_subtlex.FREQcount[df_subtlex.Word == word].values[0]
        occurences += word_count
    return occurences