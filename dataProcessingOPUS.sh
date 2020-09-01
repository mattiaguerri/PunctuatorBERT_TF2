#!/bin/bash

# path=./DataOPUS/Fr.txt
# inp=xafTrain
# out=xafTrain.pro

# path=./DataOPUS/Fr.txt/Splits
# inp=test_00
# out=test_00.pro

path=$1
inp=$2
out=$3

# Remove blank lines, convert upper case to lower case.
grep "\S" $path/$inp | tr '[:upper:]' '[:lower:]' > $path/tmp_00

# Replace group of three dots ("...") with one dot.
sed 's/\.\.\././g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove the pattern double quotes + space ("" ").
sed 's/ "//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove the pattern space + double quotes (" "").
sed 's/" //g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space + colon with colon.
sed 's/ :/:/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space + exclamation mark with exclamation mark.
sed 's/ !/!/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space + question mark with question mark.
sed 's/ ?/?/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove pattern dash + space ("- ")
sed 's/- //g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace comma, semicolon, colon, exclamation mark and question mark with period.
sed 's/,/./g' $path/tmp_00 | sed 's/;/./g' | sed 's/:/./g' | sed 's/!/./g' | sed 's/?/./g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

mv $path/tmp_00 $path/$out
