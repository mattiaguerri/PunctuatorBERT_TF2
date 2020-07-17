import numpy as np
import copy

def encodeDataTimeStamps(data, tokenizer, punctuation_enc):
    XTokensIds = []
    XTokensIdsBeg = []  
    XTokensIdsEnd = []  
    Y = []
    count = -1
    for line in data:
        count += 1
        word, punc, wordBeg, wordEnd = line.split("\t")
        tokens = tokenizer.tokenize(word)
        tokensIds = tokenizer.convert_tokens_to_ids(tokens)
        if len(tokensIds) > 0:
            ### note that one word can be encoded in more than one token
            if len(tokensIds) > 1:
                y = (len(tokensIds)-1) * [0]
                numTokens = len(tokensIds)
                for i in range(numTokens-1):
                    XTokensIdsBeg.append(float(wordBeg))
                    XTokensIdsEnd.append(float(wordEnd))
                Y += y
                # print("Line Index = ", count+1)
            XTokensIds += tokensIds
            XTokensIdsBeg.append(float(wordBeg))
            XTokensIdsEnd.append(float(wordEnd))
            Y += [punctuation_enc[punc]]
    return XTokensIds, XTokensIdsBeg, XTokensIdsEnd, Y


def correctTimeStamps(sequenceBegins, sequenceEnds, sequenceSize):
    """
    Apply two corrections to the time stamps:
        . wordEnd always larger than nextWordBegin.
        . Start time for the sequnce is zero.
    """

    ### CORRECTION 1
    ### Apply the correction to the time stamps.
    sequenceBeginsCorr = np.asarray(copy.deepcopy(sequenceBegins))
    sequenceEndsCorr = np.asarray(copy.deepcopy(sequenceEnds))
    for i in range(sequenceSize-1):
        wordBegin = sequenceBegins[i]
        wordEnd = sequenceEnds[i]
        nextWordBegin = sequenceBegins[i+1]
        nextWordEnd = sequenceEnds[i+1]
        ### i add an additional condition because sometimes wordEnd > nextWordBegin
        ### but not beacause of the start of a new sentence.
        if wordBegin != nextWordBegin and wordEnd != nextWordEnd:
            if wordEnd > nextWordBegin and abs(wordEnd - nextWordBegin) > 0.021:
                sequenceBeginsCorr[i+1:] += wordEnd
                sequenceEndsCorr[i+1:] += wordEnd
#         ### same as before but without the additional condition
#         if wordEnd > nextWordBegin:
#             sequenceBeginsCorr[i+1:] += wordEnd
#             sequenceEndsCorr[i+1:] += wordEnd
    ### CORRECTION 2
    ### Set beginning of first word in the sentence as time zero.
    sequenceBeginsCorr[:] -= sequenceBegins[0]
    sequenceEndsCorr[:] -= sequenceBegins[0]

    return list(sequenceBeginsCorr), list(sequenceEndsCorr)


def insertTargetTimeStamps(x, xBeg, xEnd, sequenceSize):
    
    X = []
    XBeg = []
    XEnd = []
    x_pad = x[-((sequenceSize-1)//2-1):]+x+x[:sequenceSize//2]
    xBeg_pad = xBeg[-((sequenceSize-1)//2-1):]+xBeg+xBeg[:sequenceSize//2]
    xEndPad = xEnd[-((sequenceSize-1)//2-1):]+xEnd+xEnd[:sequenceSize//2]

    for i in range(len(x_pad)-sequenceSize+2):
    # for i in range(1):

        ind = (sequenceSize-1)//2

        sequence = x_pad[i:i+sequenceSize-1]
        sequence.insert(ind, 0)
        X.append(sequence)

        sequenceBegs = xBeg_pad[i:i+sequenceSize-1]
        sequenceEnds = xEndPad[i:i+sequenceSize-1]
        
        # Apply corrections to the timestamps.
        sequenceBegsCorr, sequenceEndsCorr = correctTimeStamps(sequenceBegs, sequenceEnds, len(sequenceEnds))
        
        val = sequenceBegsCorr[ind-1]
        sequenceBegsCorr.insert(ind, val)
        
        val = sequenceEndsCorr[ind-1]
        sequenceEndsCorr.insert(ind, val)

        # Collect corrected data.
        XBeg.append(sequenceBegsCorr)
        XEnd.append(sequenceEndsCorr)

    return np.array(X), np.array(XBeg), np.array(XEnd)


def positionalEncoding(sequence, depth):
    
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


