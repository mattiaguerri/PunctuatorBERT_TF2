import numpy as np
import sys


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def processingIWSLT12(inpFileName):
    """
    Processing for IWSLT12 data. 
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
                outFile.write(foo[0] + '\t' + 'COMMA' + '\n')
                outFile.write(foo[1] + '\t' + 'space' + '\n')
                continue
            if splits[j][-1] == ',':
                outFile.write(splits[j][0:-1] + '\t' + 'COMMA' + '\n')
            elif splits[j][-1] == '.':
                outFile.write(splits[j][0:-1] + '\t' + 'PERIOD' + '\n')
            elif splits[j][-1] == '?':
                outFile.write(splits[j][0:-1] + '\t' + 'QUESTION' + '\n')
            else:
                outFile.write(splits[j] + '\t' + 'SPACE' + '\n')

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


def processingOPUS(inpFileName):
    """
    Processing OPUS data. 
    """
    setType = ""
    if inpFileName.find("train") != -1 or inpFileName.find("Train") != -1:
        setType = "train"
    elif inpFileName.find("valid") != -1 or inpFileName.find("Valid") != -1:
        setType = "valid"
    elif inpFileName.find("test") != -1 or inpFileName.find("Test") != -1:
        setType = "test"

    outFileName = './Data/' + setType + 'Data.txt'

    # print('\n', inpFileName)
    # print(outFileName, '\n')

    inpFile = open(inpFileName, "r")
    outFile = open(outFileName, "w")

    numLin = len(inpFile.readlines())
    inpFile.seek(0)

    for i in range(numLin):
    # for i in range(10):
        line = inpFile.readline()

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
            # check the case in which we have word, period, then another word without blank space
            if splits[j][0:-1].find(".") != -1:
                foo = splits[j].split('.')
                outFile.write(foo[0] + '\t' + 'PERIOD' + '\n')
                outFile.write(foo[1] + '\t' + 'SPACE' + '\n')
                continue
            if splits[j][-1] == '.':
                outFile.write(splits[j][0:-1] + '\t' + 'PERIOD' + '\n')
            else:
                outFile.write(splits[j] + '\t' + 'SPACE' + '\n')

    outFile.close()

    # check the output file
    inpFile = open(outFileName, "r")
    numLin = len(inpFile.readlines())
    inpFile.seek(0)
    for i in range(numLin):
        line = inpFile.readline()
        splits = line.split(',')
        if len(splits) > 2:
            print("\n\nLine:   ", i+1)
            print(splits, "\n\n")
            sys.exit()
    
    return(outFileName)


def processingIWSLT17(inpFileName):
    """
    Processing IWSLT17 French data. 
    """
    setType = ""
    if inpFileName.find("train") != -1 or inpFileName.find("Train") != -1:
        setType = "train"
    elif inpFileName.find("valid") != -1 or inpFileName.find("Valid") != -1:
        setType = "valid"
    elif inpFileName.find("test") != -1 or inpFileName.find("Test") != -1:
        setType = "test"

    outFileName = './Data/' + setType + 'Data.txt'

    # print('\n', inpFileName)
    # print(outFileName, '\n')

    inpFile = open(inpFileName, "r")
    outFile = open(outFileName, "w")

    numLin = len(inpFile.readlines())
    inpFile.seek(0)

    for i in range(numLin):
    # for i in range(2):
        line = inpFile.readline()

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
            # check if splits[j] is a number
            if isNumber(splits[j])==True:
                outFile.write(splits[j] + '\t' + 'SPACE' + '\n')
                continue
            # check if splits[j] is number+dot
            if isNumber(splits[j][0:-1])==True:
                outFile.write(splits[j][0:-1] + '\t' + 'PERIOD' + '\n')
                continue
            # check the case in which we have word+dot+word.
            if splits[j][0:-1].find(".") != -1:
                foo = splits[j].split('.')
                outFile.write(foo[0] + '\t' + 'PERIOD' + '\n')
                outFile.write(foo[1] + '\t' + 'SPACE' + '\n')
                continue
            if splits[j][-1] == '.':
                outFile.write(splits[j][0:-1] + '\t' + 'PERIOD' + '\n')
            else:
                outFile.write(splits[j] + '\t' + 'SPACE' + '\n')

    outFile.close()

    # check the output file
    inpFile = open(outFileName, "r")
    numLin = len(inpFile.readlines())
    inpFile.seek(0)
    for i in range(numLin):
        line = inpFile.readline()
        splits = line.split(',')
        if len(splits) > 2:
            print("\n\nLine:   ", i+1)
            print(splits, "\n\n")
            sys.exit()
    
    return(outFileName)


def processingScriber(inpFileName):
    """
    Processing proprietary data.
    Input a file structured in sentences.
    Output a file structured in two columns [word + \t + punctuation]. 
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

    storeWord = []
    storeWord_1 = []
    lines = inpFile.readlines()
    for line in lines:

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
                outFile.write(splits[j] + '\t' + 'SPACE' + '\n')
            elif splits[j + 1] == '<b>':
                outFile.write(splits[j] + '\t' + 'PERIOD' + '\n')

        # process the last word with the next sentence
        if splits[-1] != '<b>':
            storeWord_1.append(splits[-1])

    # insert the very last wordinpFile
    if len(storeWord_1) != 0:
        outFile.write(storeWord_1[0] + '\t' + 'SPACE' + '\n')

    outFile.close()
    inpFile.close()
    
    return(outFileName)


def encodeData(data, tokenizer, punctuation_enc):
    """
    Takes in the dataset made of two columns separated by comma.
    First column is a word, second column is what comes after the word (blank space, comma, period, etc.)
    Output X, containing the token id of the words.
    Output y, which contains what there is after the word.
    """
    X = []
    Y = []
    for line in data:
        valuesList = line.split('\t')
        word = valuesList[0]
        punc = valuesList[1].strip()
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
            if word == "":
                continue
            else:
                token = tokenizer.tokenize(word)
                tokenId = tokenizer.convert_tokens_to_ids(token)
                X += tokenId
    return X


def insertTarget(x, sequenceSize):
    """
    Restructure x in order to output a 2D array with dims (len(x), sequenceSize).
    The target is placed in the middle of the sequence.
    The target is a zero.
    
    Parameters:
    ----------
    x : list
        List of tokens ids.
    sequnceSize: integer
        Size of the sequence to be input into the model.
    
    Returns:
    X : numpy array
        2D array with dims (len(x), sequenceSize)
    """
    X = []
    x_pad = x[-((sequenceSize-1)//2-1):]+x+x[:sequenceSize//2]

    for i in range(len(x_pad)-sequenceSize+2):
        sequence = x_pad[i:i+sequenceSize-1]
        sequence.insert((sequenceSize-1)//2, 0)
        X.append(sequence)

    return np.array(X)