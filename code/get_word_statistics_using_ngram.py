import arpa
import os
import textgrids

# load general parameters
textgrid_file = '/users/spraak/mgillis/arpaModels/audiobook_1.TextGrid'
output_dir = '/users/spraak/mgillis/arpaModels/'
fs = 64 # sampling rate by preprocessed EEG of sparrkulee dataset

models = arpa.loadf("/users/spraak/mgillis/arpaModels/wiki_nl_5_gram.arpa", encoding='utf-8')
lm = models[0]

def computeWordProbability(target_dir, textgrid_file, languageModel):
    """ load the text saved in the file, preprocess this test and compute the word probaility"""

    grid = textgrids.TextGrid(textgrid_file)

    # keep the first lines except xmin and xmax => transcript remains unchanged
    f = open(os.path.join(target_dir, os.path.basename(textgrid_file)).replace('.TextGrid', '.txt'), 'w', encoding='utf-8')
    f.write('word\ttstart\ttstop\tlogWF\tsurprisal\n')

    g = open(os.path.join(target_dir, os.path.basename(textgrid_file)).replace('.TextGrid', '_info.txt'), 'w')

    previous_words = []
    n_unknown_words = 0
    for word_info in grid['ORT-MAU']:
        current_word = word_info.text
        word_start = word_info.xmin
        word_end = word_info.xmax

        if len(current_word)==0:
            # empty string → skip
            continue

        if current_word in languageModel.vocabulary():
            # get word frequency
            word_probability = -languageModel.log_p(current_word)

            # get word surprisal
            word_surprisal = -languageModel.log_p(previous_words + [current_word])

            previous_words.append(current_word)
            if len(previous_words) > 5:
                previous_words = previous_words[1::]

            words_line = [current_word, word_start, word_end, word_probability, word_surprisal]
            f.write('\t'.join(['{:.3f}'.format(i) if type(i) == float else str(i) for i in words_line]) + '\n')

        else:
            g.write('word not recognized: %s \n' % current_word)
            n_unknown_words += 1
            continue
    f.close()
    g.write('amount of unknown words: %d \n' % n_unknown_words)
    g.close()


if __name__ == '__main__':
    computeWordProbability(output_dir, textgrid_file, lm)
