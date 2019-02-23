from voc import Voc
from data_preprocess import unicodeToAscii, normalizeString, extractSentencePairs,\
    loadPrepareData, trimRareWords,printsample
import torch

if __name__=="__main__":

    voc, qa_pairs = loadPrepareData("QQbot", "formatted.txt")

    printsample(qa_pairs)

    # print(torch.mean(torch.Tensor(list(map(len,sentences))))) ##tensor(9.0420)
    # mean length of sentence is 9.0420, then let the max_length become 15

    # Trim voc and pairs
    pairs = trimRareWords(voc, qa_pairs)

    printsample(pairs)