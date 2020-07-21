import numpy as np
import copy


def loadFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


def encodeDataTimeStamps(data, tokenizer, punctuationEnc):
    """
    Encode the data.
    Take the words and encode them in tokens ids.
    
    Parameters:
    ----------
    data : list
        Every element of the list is a line of the input file.
    tokenizer : transformers.tokenization
        The tokenizer.
    punctuationEnc : dict
        Keys are punctuation type, values are integers.
    
    Return:
    ------
    XTokensIds, XTokensIdsBeg, XTokensIdsEnd, XTokensIdsGap, Y : 1D numpy arrays
        1D arrays, the dimension is equal to the number of tokens in the input file.
    """
    
    XTokensIds = []
    XTokensIdsBeg = []  
    XTokensIdsEnd = []
    XTokensIdsGap = []
    Y = []
    count = -1
    for line in data:
        count += 1
        items = line.split("\t")
        word = items[0].strip()
        punc = items[1].strip()
        wordBeg = float(items[2])
        wordEnd = float(items[3])
        gap = float(items[4])
        tokens = tokenizer.tokenize(word)
        tokensIds = tokenizer.convert_tokens_to_ids(tokens)
        if len(tokensIds) > 0:
            ### note that one word can be encoded in more than one token
            if len(tokensIds) > 1:
                numTokens = len(tokensIds)
                y = (numTokens-1) * [0]
                for i in range(numTokens-1):
                    XTokensIdsBeg.append(wordBeg)
                    XTokensIdsEnd.append(wordEnd)
                    XTokensIdsGap.append("0.0000")
                Y += y
            XTokensIds += tokensIds
            XTokensIdsBeg.append(wordBeg)
            XTokensIdsEnd.append(wordEnd)
            XTokensIdsGap.append(gap)
            Y += [punctuationEnc[punc]]
    return np.asarray(XTokensIds), np.asarray(XTokensIdsBeg), np.asarray(XTokensIdsEnd), np.asarray(XTokensIdsGap), np.asarray(Y)


def correctTimeStamps(sequenceBeg, sequenceEnd, sequenceSize):
    """
    Apply two corrections to the time stamps:
        . nextWordBegin has to be always larger than wordEnd.
        . Set start time for each sequence at zero.
    """

    sequenceBegCor = np.asarray(copy.deepcopy(sequenceBeg))
    sequenceEndCor = np.asarray(copy.deepcopy(sequenceEnd))

    ### CORRECTION 1
    ### I want nextWordBeg always bggr than wordEnd
    for i in range(sequenceSize-1):
        wordBegin = sequenceBeg[i]
        wordEnd = sequenceEnd[i]
        nextWordBegin = sequenceBeg[i+1]
        nextWordEnd = sequenceEnd[i+1]
        if wordEnd > nextWordBegin:
            # if nextWordBegin != .0:
                # print("WARNING: nextWordBegin = ", nextWordBegin)
                # print("Line Index = ", i+1)
                # print(wordBegin, "   ", wordEnd, "\n")
            sequenceBegCor[i+1:] += wordEnd
            sequenceEndCor[i+1:] += wordEnd
            
#     ### CORRECTION 2
#     ### Set beginning of the sequence to zero.
#     sequenceBegCor[:] -= sequenceBeg[0]
#     sequenceEndCor[:] -= sequenceBeg[0]

    return list(sequenceBegCor), list(sequenceEndCor)


def insertTargetTimeStamps(x, xBeg, xEnd, xGap, sequenceSize):
    """
    Take in 1D array with dimension equal to number of tokens ids.
    Output 2D arrays where for each token ids there is a sequence with the target inserted.
    
    Parameters:
    ----------
    x, xBeg, xEnd, xGap : 1D numpy arrays
        Arrays of dimension equal to the number of tokens ids.
    sequenceSize : integer
        The size of the output sequence.
    
    Return:
    X, XBeg, XEnd, XGap : 2D numpy arrays
        Arrays of dimension (number tokens ids, sequence size)
    """
    
    x = list(x)
    xBeg = list(xBeg)
    xEnd = list(xEnd)
    xGap = list(xGap)

    X = []
    XBeg = []
    XEnd = []
    XGap = []

    xPad = x[-((sequenceSize-1)//2-1):] + x + x[:sequenceSize//2]
    xBegPad = xBeg[-((sequenceSize-1)//2-1):] + xBeg + xBeg[:sequenceSize//2]
    xEndPad = xEnd[-((sequenceSize-1)//2-1):] + xEnd + xEnd[:sequenceSize//2]
    xGapPad = xGap[-((sequenceSize-1)//2-1):] + xGap + xGap[:sequenceSize//2]

    for i in range(len(xPad)-sequenceSize+2):
    # for i in range(1):

        ind = (sequenceSize-1)//2

        sequence = xPad[i:i+sequenceSize-1]
        sequence.insert(ind, 0)
        X.append(sequence)

        sequenceBeg = xBegPad[i:i+sequenceSize-1]
        sequenceEnd = xEndPad[i:i+sequenceSize-1]
        sequenceGap = xGapPad[i:i+sequenceSize-1]

        # Apply corrections to the timestamps.
        sequenceBegCor, sequenceEndCor = correctTimeStamps(sequenceBeg, sequenceEnd, len(sequenceBeg))

        val = sequenceBegCor[ind-1]
        sequenceBegCor.insert(ind, val)

        val = sequenceEndCor[ind-1]
        sequenceEndCor.insert(ind, val)

        # insert the gap for the target
        sequenceGap.insert(ind, "0.0000")

        # Collect corrected data.
        XBeg.append(sequenceBegCor)
        XEnd.append(sequenceEndCor)
        XGap.append(sequenceGap)

    return np.array(X), np.array(XBeg), np.array(XEnd), np.array(XGap)


def positionalEncoding(sequence, depth):
    """
    Compute positional encoding vector as in
    https://github.com/tensorflow/examples/blob/master/community/en/position_encoding.ipynb
    
    Parameters:
    ----------
    sequence : 2D array-like
        2D Array with dimensions (batchSize, sequenceSize). It contains the elemnts 
        to be encoded, it could be, for example, positions or time-stamps.
    depth : integer
        Dimension of the encoding vector.
    
    Return:
    ------
    out : 3D numpy array
        Array containing the encoded information. Dimensions are (batchSize, sequenceSize, depth).
    
    """
    
    batchSize = sequence.shape[0]
    sequenceSize = sequence.shape[1]
    
    min_rate = 1/10000

    assert depth%2 == 0, "Depth must be even."
    angle_rate_exponents = np.linspace(0,1,depth//2)
    angle_rates = min_rate**(angle_rate_exponents)
    
    angle_rads = sequence[:, :, np.newaxis]*angle_rates[np.newaxis, np.newaxis, :]

    out = np.empty((batchSize, sequenceSize, depth))
    for i in range(batchSize):
        sines = np.sin(angle_rads[i, :, :])
        cosines = np.cos(angle_rads[i, :, :])
        arr = np.reshape(np.vstack((sines, cosines)).ravel('F'), (sequenceSize, depth), order='F')
        out[i, :, :] = arr
    
    return out
