# encoding=utf-8

import re
import torch
import itertools
import jieba
import unicodedata
from voc import Voc
from values import MAX_LENGTH, MIN_COUNT


PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)

def printsample(qa_pairs):
    print("\nsome sample pairs:")
    for pair in qa_pairs[:5]:
        print(pair)
    print("\n")


def extractSentencePairs(conversations):
    qa_pairs = []
    for i in range(len(conversations) - 1):
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        inputLine = conversations[i]
        targetLine = conversations[i + 1]
        # Filter wrong samples (if one of the lists is empty)
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Turn a unicode string to plain ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def filterPair(p, max_length=MAX_LENGTH):
    # Input sequences need to preserve the last word for EOS token
    if len(p) > 0:
        return len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH
    else:
        return None


# Filter pairs using filterPair condition
def filterPairs(pairs, max_lenth=MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, max_lenth)]


def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    # Split every line into pairs and normalize
    sentences = []
    pairs = []
    with open(datafile, "r") as file:
        for line in file.readlines():
            line = line.strip()
            line = line.replace('\n', "")
            sentences.append(line)
    for i in range(len(sentences) - 1):  # We ignore the last line (no answer for it)
        inputLine = sentences[i]
        targetLine = sentences[i + 1]
        # Filter wrong samples (if one of the lists is empty)
        if inputLine and targetLine:
            pairs.append([inputLine, targetLine])
    voc = Voc(corpus_name)
    return voc, pairs


def loadPrepareData(corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
    voc.addSentence(pairs[len(pairs) - 1][1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, min_count=MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in jieba.cut(input_sentence, cut_all=False):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in jieba.cut(output_sentence, cut_all=False):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in jieba.cut(sentence,cut_all=False)] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
