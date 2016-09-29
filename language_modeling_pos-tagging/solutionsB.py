from __future__ import division
import sys
import nltk
import math
import time
from collections import defaultdict, Counter

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sent in brown_train:
        sent_tags = []
        sent_tokens = []
        new_sent = sent.split()
        new_sent.insert(0, START_SYMBOL)
        new_sent.insert(0, START_SYMBOL)
        new_sent.append(STOP_SYMBOL)
        for token_tag in new_sent:
            pair = token_tag.rsplit('/', 1)
            sent_tokens.append(pair[0])
            if len(pair) > 1:
                sent_tags.append(pair[1])
            else:
                sent_tags.append(pair[0])
        brown_words.append(sent_tokens)
        brown_tags.append(sent_tags)
    return brown_words, brown_tags


# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_list = []
    trigram_list = []
    bigram_counts = defaultdict(int)
    trigram_counts = defaultdict(int)

    for sent in brown_tags:
        sent_trigrams = zip(sent, sent[1:], sent[2:])
        trigram_list += sent_trigrams

        sent_bigrams = zip(sent, sent[1:])
        bigram_list += sent_bigrams

    for bigram in bigram_list:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    for trigram in trigram_list:
        trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1

    for trigram in trigram_counts:
        q_values[trigram] = math.log(trigram_counts[trigram] / bigram_counts[(trigram[0], trigram[1])], 2)
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    word_counts = Counter([word for sent in brown_words for word in sent])
    for word in word_counts:
        if word_counts[word] > 5:
            known_words.add(word)
    return known_words

# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sent in brown_words:
        sent_words_rare = []
        for word in sent:
            if word not in known_words:
                word = RARE_SYMBOL
            sent_words_rare.append(word)
        brown_words_rare.append(sent_words_rare)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    word_tag_counts = defaultdict(int)
    word_counts = Counter([word for sent in brown_words_rare for word in sent])
    tag_counts =  Counter([tag for sent in brown_tags for tag in sent])

    for i, sent in enumerate(brown_words_rare):
        for j, word in enumerate(sent):
            word_tag_counts[(word, brown_tags[i][j])] = word_tag_counts.get((word, brown_tags[i][j]), 0) + 1

    for word_tag in word_tag_counts:
        # word_count = sum(sent.count(word_tag[0]) for sent in brown_words_rare)
        e_values[word_tag] = math.log(word_tag_counts[word_tag] / tag_counts[word_tag[1]], 2)
        taglist.add(word_tag[1])
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!


def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    best_score = {}
    for sent in brown_dev_words:
        tagged_sent = ""
        prev_score = best_score["0 * *"] = 0
        grandparent = START_SYMBOL
        parent = START_SYMBOL
        for i, word in enumerate(sent):
            if word in known_words:
                max = None
                for tag in taglist:
                    if (word, tag) in e_values and (grandparent, parent, tag) in q_values:
                        score = prev_score + q_values[(grandparent, parent, tag)] + e_values[(word, tag)]
                        if max is None or score > max:
                            max = score
                            max_tag = tag
                grandparent = parent
                parent = max_tag
                if max is None:
                    tagged_sent += word + "/" + "NOUN "
                else:
                    prev_score = max
                    best_score[str(i+1) + " " + grandparent + " " + parent] = max
                    tagged_sent += word + "/" + parent + " "
            else:
                tagged_sent += word + "/" + "NOUN "
        tagged_sent += "\n"
        tagged.append(tagged_sent)
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for sent in brown_dev_words:
        tagged_sentence = trigram_tagger.tag(sent)
        new_sent = ' '.join(map('/'.join,tagged_sentence))
        tagged.append(new_sent + "\n")
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
