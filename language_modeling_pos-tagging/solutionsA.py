from __future__ import division
import math
import nltk
import time
from collections import defaultdict


# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    corpus = []

    unigram_counts = defaultdict(int)
    total_unigrams = 0

    bigram_list = []
    bigram_counts = defaultdict(int)

    trigram_list = []
    tri_bigram_list = []
    tri_bigram_counts = defaultdict(int)
    trigram_counts = defaultdict(int)

    for sent in training_corpus:
        new_sent = sent.split()
        new_sent.insert(0, START_SYMBOL)
        new_sent.append(STOP_SYMBOL)
        corpus.append(new_sent)

    for sent in corpus:
        # unigrams
        for i, token in enumerate(sent):
            unigram_counts[token] = unigram_counts.get(token, 0) + 1
            if token != START_SYMBOL:
                total_unigrams += 1

        # bigrams
        sent_bigrams = zip(sent, sent[1:])
        bigram_list += sent_bigrams

        # trigrams
        sent.insert(0, START_SYMBOL)

        tri_bigrams = zip(sent, sent[1:])
        tri_bigram_list += tri_bigrams

        sent_trigrams = zip(sent, sent[1:], sent[2:])
        trigram_list += sent_trigrams



    # unigrams
    for unigram in unigram_counts:
        unigram_p[(unigram)] = math.log(unigram_counts[unigram] / total_unigrams, 2)
    print unigram_p[("captain")]
    print unigram_p[("captain's")]
    print unigram_p[("captaincy")]


    # bigrams
    for bigram in bigram_list:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    for bigram in bigram_counts:
        bigram_p[bigram] = math.log(bigram_counts[bigram] / unigram_counts[bigram[0]], 2)
    print bigram_p[("and", "religion")]
    print bigram_p[("and", "religious")]
    print bigram_p[("and", "religiously")]

    # trigrams

    for bigram in tri_bigram_list:
        tri_bigram_counts[bigram] = tri_bigram_counts.get(bigram, 0) + 1

    for trigram in trigram_list:
        trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1

    for trigram in trigram_counts:
        trigram_p[trigram] = math.log(trigram_counts[trigram] / tri_bigram_counts[(trigram[0], trigram[1])], 2)
    print trigram_p[("and", "not", "a")]
    print trigram_p[("and", "not", "by")]
    print trigram_p[("and", "not", "come")]
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    new_corpus = []
    for sent in corpus:
        new_sent = sent.split()
        new_sent.insert(0, START_SYMBOL)
        new_sent.append(STOP_SYMBOL)
        new_corpus.append(new_sent)

    for sent in new_corpus:
        if n == 1:
            try:
                score = sum([ngram_p[token] for token in sent[1:]])
            except:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
            scores.append(score)
        elif n == 2:
            bigram_tuples = list(nltk.bigrams(sent))
            try:
                score = sum([ngram_p[bigram] for bigram in bigram_tuples])
            except:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
            scores.append(score)
        elif n == 3:
            sent.insert(0, START_SYMBOL)
            trigram_tuples = list(nltk.trigrams(sent))
            score = 0
            for trigram in trigram_tuples:
                    try:
                        score += ngram_p[trigram]
                    except:
                        score = MINUS_INFINITY_SENTENCE_LOG_PROB
            scores.append(score)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    new_corpus = []
    for sent in corpus:
        new_sent = sent.split()
        new_sent.insert(0, START_SYMBOL)
        new_sent.insert(0, START_SYMBOL)
        new_sent.append(STOP_SYMBOL)
        new_corpus.append(new_sent)
    lambd = 1.0/3
    for sent in new_corpus:
        score = 0
        trigram_tuples = list(nltk.trigrams(sent))
        for trigram_tuple in trigram_tuples:
            if trigram_tuple in trigrams:
                # score += 2 ** (trigrams[trigram_tuple]) / 3.0
                score += trigrams[trigram_tuple] * lambd
            if (trigram_tuple[0], trigram_tuple[1]) in bigrams:
                # score += 2 ** (bigrams[(trigram_tuple[0], trigram_tuple[1])]) / 3.0
                score += bigrams[(trigram_tuple[0], trigram_tuple[1])] * lambd
            if (trigram_tuple[0]) in unigrams:
                # score += 2 ** (unigrams[(trigram_tuple[0])]) / 3.0
                score += unigrams[(trigram_tuple[0])] * lambd
        # for i, word in enumerate(sent[2:]):
        #     print word
        #     score += math.log(2 ** (unigrams[word]) / 3.0 + 2 ** (bigrams[(sent[i-1], sent[i])]) / 3.0 + 2 ** (trigrams[(sent[i-2], sent[i-1], sent[i])]) / 3.0, 2)

        # scores.append(math.log(score))
        scores.append(score)

        # log(2 ^ (unigram_p) / 3.0 + 2 ^ (bigram_p) / 3.0 + 2 ^ (trigram_p) / 3.0, 2)
        #
        # for each word, then sum them up for the sentence.

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
