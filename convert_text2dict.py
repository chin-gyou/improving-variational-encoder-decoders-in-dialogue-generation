import collections
import numpy
import operator
import os
import sys
import logging
import cPickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text2dict')

def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)
    
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Dialogue file; assumed shuffled with one document (e.g. one movie dialogue, or one Twitter conversation or one Ubuntu conversation) per line")
parser.add_argument("--cutoff", type=int, default=-1, help="Vocabulary cutoff (optional)")
parser.add_argument("--dict", type=str, default="", help="External dictionary (pkl file)")
parser.add_argument("output", type=str, help="Prefix of the pickle binarized dialogue corpus")
args = parser.parse_args()

if not os.path.isfile(args.input):
    raise Exception("Input file not found!")

unk = "<unk>"

###############################
# Part I: Create the dictionary
###############################
if args.dict != "":
    # Load external dictionary
    assert os.path.isfile(args.dict)
    vocab = dict([(x[0], x[1]) for x in cPickle.load(open(args.dict, "r"))])
    
    # Check consistency
    assert '<unk>' in vocab
    assert '__eou__' in vocab
else:
    temp_dic = {}
    for line in open(args.input, 'r'):
        for b in line.strip().split():
            temp_dic.setdefault(b.lower(),0)
            temp_dic[b.lower()] +=1
    li = sorted(temp_dic.iteritems(), key=lambda d:d[1], reverse = True)
    if args.cutoff != -1:
        logger.info("Cutoff %d" % args.cutoff)
        vocab_count = [word[0] for fre, word in enumerate(li) if fre <= args.cutoff]
            
    else:
        vocab_count = li

    # Add special tokens to the vocabulary
    vocab = {'<unk>': 0, '__eou__': 1}

    # Add other tokens to vocabulary in the order of their frequency
    i = 2
    for word in vocab_count:
        if not word in vocab:
            vocab[word] = i
            i += 1

logger.info("Vocab size %d" % len(vocab))

#################################
# Part II: Binarize the dialogues
#################################

# Everything is loaded into memory for the moment
binarized_corpus = []
# Some statistics
unknowns = 0.
num_terms = 0.
freqs = collections.defaultdict(lambda: 0)

# counts the number of dialogues each unique word exists in; also known as document frequency
df = collections.defaultdict(lambda: 0)

for line, dialogue in enumerate(open(args.input, 'r')):
    dialogue_words = dialogue.strip().split()
    if dialogue_words[len(dialogue_words)-1] != '__eou__':
        dialogue_words.append('__eou__')

    # Convert words to token ids and compute some statistics
    dialogue_word_ids = []
    for word in dialogue_words:
        word_id = vocab.get(word.lower(), 0)
#        if word_id == 0:
#            print word
        dialogue_word_ids.append(word_id)
        unknowns += 1 * (word_id == 0)
        freqs[word_id] += 1

    num_terms += len(dialogue_words)

    # Compute document frequency statistics
    unique_word_indices = set(dialogue_word_ids)
    for word_id in unique_word_indices:
        df[word_id] += 1

    # Add dialogue to corpus
    binarized_corpus.append(dialogue_word_ids)

safe_pickle(binarized_corpus, args.output + ".dialogues.pkl")

if args.dict == "":
     safe_pickle([(word, word_id, freqs[word_id], df[word_id]) for word, word_id in vocab.items()], args.output + ".dict.pkl")

logger.info("Number of unknowns %d" % unknowns)
logger.info("Number of terms %d" % num_terms)
logger.info("Mean document length %f" % float(sum(map(len, binarized_corpus))/len(binarized_corpus)))
logger.info("Writing training %d dialogues (%d left out)" % (len(binarized_corpus), line + 1 - len(binarized_corpus)))
