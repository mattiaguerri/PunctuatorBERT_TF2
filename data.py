
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys


def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


def encode_data(data, tokenizer, punctuation_enc):
    """
    Converts words to (BERT) tokens and punctuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for line in data:
        word, punc = line.split(',')
        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y


def insert_target(x, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.
    """
    X = []
    x_pad = x[-((segment_size-1)//2-1):]+x+x[:segment_size//2]

    for i in range(len(x_pad)-segment_size+2):
        segment = x_pad[i:i+segment_size-1]
        segment.insert((segment_size-1)//2, 0)
        X.append(segment)

    return np.array(X)


def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    X, y = encode_data(data, tokenizer, punctuation_enc)
    X = insert_target(X, segment_size)
    return X, y


def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def preProcessing(inpFileName):

    setType = ""
    if inpFileName.find("Train") != -1:
        setType = "train"
    elif inpFileName.find("Valid") != -1:
        setType = "valid"
    elif inpFileName.find("Test") != -1:
        setType = "test"

    outFileName = './Data/' + setType + 'Set_02.txt'

    print('\n\n', inpFileName)
    print(outFileName, '\n\n')

    inpFile = open(inpFileName, "r")
    outFile = open(outFileName, "w")

    numLin = len(inpFile.readlines())
    inpFile.seek(0)

    for i in range(numLin):
        line = inpFile.readline()

        # print('\n', i + 1)

        splits = line.split(' ')
        # print(splits)

        # remove the new line character
        if splits[-1] == '\n':
            splits = splits[0:-1]
        # remove new line char in case is contained in the last element
        if splits[-1][-1] == '\n':
            splits[-1] = splits[-1][0:-1]

        # remove empty elements
        while "" in splits:
            splits.remove("")

        # if splits only has one element
        if len(splits) == 1:

            if splits[0][-1] != ',' and splits[0][-1] != '.' and splits[0][-1] != '?':
                outFile.write(splits[0] + ',O' + '\n')
            elif splits[0][-1] == ',':
                outFile.write(splits[0][0:-1] + ',COMMA' + '\n')
            elif splits[0][-1] == '.':
                outFile.write(splits[0][0:-1] + ',PERIOD' + '\n')
            elif splits[0][-1] == '?':
                outFile.write(splits[0][0:-1] + ',QUESTION' + '\n')
            continue

        # if splits has more than one element
        else:
            for j in range(len(splits)):

                # you could have something like this: 'car,engine'
                if splits[j][0:-1].find(',') != -1:
                    subSplits = splits[j].split(',')
                    outFile.write(subSplits[0] + ',COMMA' + '\n')
                    outFile.write(subSplits[1] + ',O' + '\n')
                    continue

                if splits[j][-1] != ',' and splits[j][-1] != '.' and splits[j][-1] != '?':
                    outFile.write(splits[j] + ',O' + '\n')
                elif splits[j][-1] == ',':
                    outFile.write(splits[j][0:-1] + ',COMMA' + '\n')
                elif splits[j][-1] == '.':
                    outFile.write(splits[j][0:-1] + ',PERIOD' + '\n')
                elif splits[j][-1] == '?':
                    outFile.write(splits[j][0:-1] + ',QUESTION' + '\n')
    outFile.close()

    # Check the output.

    inpFile = open(outFileName, "r")

    numLin = len(inpFile.readlines())
    inpFile.seek(0)

    for i in range(numLin):
        line = inpFile.readline()
        splits = line.split(',')

        if len(splits) != 2:
            print('\n\n\nERROR!!!')
            print('\nLINE: ..... ', i + 1)
            print(splits)
            break














