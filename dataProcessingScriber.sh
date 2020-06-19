#!/bin/bash

path=./DataScriber
inp=scriberTrain
out=scriberTrainPro

# Remove blank lines.
grep "\S" $path/$inp > $path/tmp_00

# Remove parenthesis. Remove space+space.
sed "s/[(][^)]*[)]//g" $path/tmp_00 |  sed 's/  / /g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Correct space+symbol
sed 's/ \././g' $path/tmp_00 | sed 's/ ,/,/g' | sed 's/ ;/;/g' | sed 's/ :/:/g' | sed 's/ !/!/g' | sed 's/ ?/?/g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove slash+slash
sed 's/\/\///g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove quotes.
sed 's/"//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace comma, semicolon, colon, exclamation mark and question mark with dot.
sed 's/,/./g' $path/tmp_00 | sed 's/;/./g' | sed 's/:/./g' | sed 's/!/./g' | sed 's/?/./g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Convert upper case to lower case.
cat $path/tmp_00 | tr '[:upper:]' '[:lower:]' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Remove blank lines.
grep "\S" $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space+space with space.
sed 's/  / /g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

mv $path/tmp_00 $path/$out
