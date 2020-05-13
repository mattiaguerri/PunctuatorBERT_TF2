import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt
import sys


# def plot_batch_sizes(ds):
    # batch_sizes = [batch.shape[0] for batch in ds]
    # plt.bar(range(len(batch_sizes)), batch_sizes)
    # plt.xlabel('Batch number')
    # plt.ylabel('Batch size')


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


def encode_data(data, tokenizer, punctuation_enc):
    """
    Takes in the dataset made of two columns separated by comma.
    First column is a word, second column is what comes after the word (blank space, comma, period, etc.)
    Output X, containing the token id of the words.
    Output y, which contains what there is after the word.
    """
    X = []
    Y = []
    for line in data:
        word, punc = line.split(',')
        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        # one word can be encoded in more than one token
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y


def encodeDataInfer(data, tokenizer):
    """
    Takes in the data made of sentences, with words separated by blank spaces (no punctuation).
    Output X, list containing the token id of the words.
    """
    X = []
    for line in data:
        words = line.split(" ")
        words[-1] = words[-1].strip()  # get rid of \n at the end of the line
        for word in words:
            token = tokenizer.tokenize(word)
            tokenId = tokenizer.convert_tokens_to_ids(token)
            X += tokenId
    return X


def insert_target(x, segment_size):
    """
    Restructure x in order to have sequences of length equal to segment_size.
    Output X, an array with dimensions len(x) * segment_size.
    In each segment the target is placed in the middle and replaced with a zero.
    """
    X = []
    x_pad = x[-((segment_size-1)//2-1):]+x+x[:segment_size//2]

    for i in range(len(x_pad)-segment_size+2):
        segment = x_pad[i:i+segment_size-1]
        segment.insert((segment_size-1)//2, 0)
        X.append(segment)

    return np.array(X)


# def process_data(data, tokenizer, punctuation_enc, segment_size):
#     X, y = encode_data(data, tokenizer, punctuation_enc)
#     X = insert_target(X, segment_size)
#     return X, y


def preProcessingScriber(inpFileName):
    """
    Preprocessing for proprietory data. 
    """
    setType = ""
    if inpFileName.find("Train") != -1:
        setType = "train"
    elif inpFileName.find("Valid") != -1:
        setType = "valid"
    elif inpFileName.find("Test") != -1:
        setType = "test"

    outFileName = './Data/' + setType + 'Set_02.txt'

    # print('\n\n', inpFileName)
    # print(outFileName, '\n\n')

    inpFile = open(inpFileName, "r")
    outFile = open(outFileName, "w")

    numLin = len(inpFile.readlines())
    inpFile.seek(0)

    storeWord = []
    storeWord_1 = []
    for i in range(numLin):
        line = inpFile.readline()

        # print('\n', i + 1)

        splits = line.split(' ')

        # print(splits)

        # remove empty elements
        while "" in splits:
            splits.remove("")

        # splits might contain only one element, which is '\n'
        if len(splits) == 1 and splits[0] == '\n':
            continue

        # remove the new line character
        if splits[-1] == '\n':
            splits = splits[0:-1]
        # remove new line char in case is contained in the last element
        if splits[-1][-1] == '\n':
            splits[-1] = splits[-1][0:-1]

        # in case previous splits was only one word, insert that word
        if len(storeWord) != 0:
            splits.insert(0, storeWord[0])
            storeWord = []

        # at this point, you could have splits with only one word
        # store the word and use it in the next sentence.
        if len(splits) == 1 and splits[0] != '\n':
            storeWord.insert(0, splits[0])
            continue

        # if splits only has less than 2 elements:
        if len(splits) < 2:
            print('\n\n !!! splits only has one element. !!! \n\n')
            sys.exit()

        # write the output
        # introduce first a possible word from previous sentence
        if len(storeWord_1) != 0:
            splits.insert(0, storeWord_1[0])
            storeWord_1 = []
        for j in range(len(splits) - 1):
            if splits[j] == '<b>':
                continue
            if splits[j + 1] != '<b>':
                outFile.write(splits[j] + ',' + 'O' + '\n')
            elif splits[j + 1] == '<b>':
                outFile.write(splits[j] + ',' + 'PERIOD' + '\n')

        # process the last word with the next sentence
        if splits[-1] != '<b>':
            storeWord_1.append(splits[-1])

    # insert the very last word
    if len(storeWord_1) != 0:
        outFile.write(storeWord_1[0] + ',' + 'O' + '\n')

    outFile.close()
    
    return(outFileName)


def preProcessingIWSLT12(inpFileName):
    """
    Preprocessing for IWSLT12 data. 
    """
    setType = ""
    if inpFileName.find("Train") != -1:
        setType = "train"
    elif inpFileName.find("Valid") != -1:
        setType = "valid"
    elif inpFileName.find("Test") != -1:
        setType = "test"

    outFileName = './Data/' + setType + 'Set_02.txt'

    # print('\n', inpFileName)
    # print(outFileName, '\n')

    inpFile = open(inpFileName, "r")
    outFile = open(outFileName, "w")

    numLin = len(inpFile.readlines())
    inpFile.seek(0)

    for i in range(numLin):
    # for i in range(2):
        line = inpFile.readline()

        # print('\n', i + 1)

        splits = line.split(' ')

        # remove empty elements
        while "" in splits:
            splits.remove("")

        # splits might contain only one element, which is '\n'
        if len(splits) == 1 and splits[0] == '\n':
            continue

        # remove the new line character
        if splits[-1] == '\n':
            splits = splits[0:-1]
        # remove new line char in case is contained in the last element
        if splits[-1][-1] == '\n':
            splits[-1] = splits[-1][0:-1]

        # write the output
        for j in range(len(splits)):
            # check the case in which we have word, comma, then another word without blank space
            if splits[j][0:-1].find(",") != -1:
                foo = splits[j].split(',')
                outFile.write(foo[0] + ',' + 'COMMA' + '\n')
                outFile.write(foo[1] + ',' + 'O' + '\n')
                continue
            if splits[j][-1] == ',':
                outFile.write(splits[j][0:-1] + ',' + 'COMMA' + '\n')
            elif splits[j][-1] == '.':
                outFile.write(splits[j][0:-1] + ',' + 'PERIOD' + '\n')
            elif splits[j][-1] == '?':
                outFile.write(splits[j][0:-1] + ',' + 'QUESTION' + '\n')
            else:
                outFile.write(splits[j] + ',' + 'O' + '\n')

    outFile.close()

    # # check the output file
    # inpFile = open(outFileName, "r")
    # numLin = len(inpFile.readlines())
    # inpFile.seek(0)
    # for i in range(numLin):
        # line = inpFile.readline()
        # splits = line.split(',')
        # if len(splits) > 2:
            # print("\n\nLine:   ", i+1)
            # print(splits, "\n\n")
            # sys.exit()
    
    return(outFileName)

