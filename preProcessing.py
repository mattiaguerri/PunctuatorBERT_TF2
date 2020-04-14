import sys

path = 'Data/'
name = 'extract'
typeData = 'Test'

inpFileName = name + typeData + '_01' + '.txt'
outFileName = name + typeData + '_02' + '.txt'

inpFile = open(path+inpFileName, "r")
outFile = open(path+outFileName, "w")

numLin = len(inpFile.readlines())
inpFile.seek(0)

for i in range(numLin):
    line = inpFile.readline()

    print('\n', i+1)

    splits = line.split(' ')
    print(splits)

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

inpFile = open(path + outFileName, "r")

numLin = len(inpFile.readlines())
inpFile.seek(0)

for i in range(numLin):
    line = inpFile.readline()
    splits = line.split(',')

    if len(splits) != 2:
        print('\n\n\nERROR!!!')
        print('\nLINE: ..... ', i+1)
        print(splits)
        break
