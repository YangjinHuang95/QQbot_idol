from data_preprocess import loadPrepareData, trimRareWords,\
    printsample, batch2TrainData,indexesFromSentence
import random
import jieba


if __name__=="__main__":

    voc, qa_pairs = loadPrepareData("QQbot", "formatted.txt")

    printsample(qa_pairs)

    # print(torch.mean(torch.Tensor(list(map(len,sentences))))) ##tensor(9.0420)
    # mean length of sentence is 9.0420, then let the max_length become 15

    # Trim voc and pairs
    pairs = trimRareWords(voc, qa_pairs)

    for i in range(100,110):
        print(voc.index2word[i])
    print("\n")

    printsample(pairs)

    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

    inp = []
    out = []
    for pair in pairs[:10]:
        inp.append(pair[0])
        out.append(pair[1])
    print(inp)
    print(len(inp))
    wordsplit = [list(jieba.cut(sentence,cut_all=False)) for sentence in inp]
    indexes = [indexesFromSentence(voc,sentence) for sentence in inp]
    print(wordsplit)
    print(indexes)

