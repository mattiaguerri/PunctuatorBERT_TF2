#!/bin/bash

# path=./DataIWSLT17/en-fr
# inp=train.fr.toy
# out=train.fr.toy.pro

path=./DataIWSLT17/en-fr
inp=train.fr
out=train.fr.pro

# Remove blank lines.
grep "\S" $path/$inp > $path/tmp_00

# Remove quotes.
sed 's/"//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space+dash+space wit dot+space.
sed 's/ - /\. /g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space+colon+space+virgolette with dot.
sed 's/ : «/\./g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space+virgolette with dot.
sed 's/ «/\./g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace dot+space+virgolette with dot.
sed 's/\. »/\./g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove space+dash+dash.
sed 's/ --//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Correct space+colon space+exclamationmark space+questionmark
sed 's/ :/:/g' $path/tmp_00 | sed 's/ !/!/g' | sed 's/ ?/?/g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove parenthesis.
sed 's/\[//g' $path/tmp_00 | sed 's/\]//g' | sed 's/(//g' | sed 's/)//g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove dash+dot. Remove space+dash+space.
sed 's/-\.//g' $path/tmp_00 | sed 's/ - //g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace dot+dot+dot with dot.
sed 's/\.\.\././g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace dot+dot with dot.
sed 's/\.\././g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace comma, semicolon, colon, exclamation mark and question mark with period.
sed 's/,/./g' $path/tmp_00 | sed 's/;/./g' | sed 's/:/./g' | sed 's/!/./g' | sed 's/?/./g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space+dot with space. Replace space+dash with space.
sed 's/ \./ /g' $path/tmp_00 | sed 's/ -/ /g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Convert upper case to lower case.
cat $path/tmp_00 | tr '[:upper:]' '[:lower:]' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

mv $path/tmp_00 $path/$out
