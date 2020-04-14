#!/bin/bash

path=./DataScriber

#inp=IWSLT12.TALK.train.en.txt.All_00
#out=IWSLT12.TALK.train.en.txt.All_01

inp=raw.processed_00.txt
out=raw.processed_01.txt

# Remove blank lines, remove quotes, convert upper case to lower case.
grep "\S" $path/$inp | tr -d '"'| tr '[:upper:]' '[:lower:]' > $path/tmp.txt

# Replace colon with dot.
sed -r 's/\:/./g' $path/tmp.txt > $path/tmptmp.txt && mv $path/tmptmp.txt $path/tmp.txt
# Replace semicolon with dot.
sed 's/\;/\./g' $path/tmp.txt > $path/tmptmp.txt && mv $path/tmptmp.txt $path/tmp.txt
# Replace exclamation mark with dot.
sed 's/\!/\./g' $path/tmp.txt > $path/tmptmp.txt && mv $path/tmptmp.txt $path/tmp.txt
# # Replace parenthesis with commas.
# sed 's/\(/\,/g' $path/tmp.txt > $path/tmptmp.txt && mv $path/tmptmp.txt $path/tmp.txt
# sed 's/\)/\,/g' $path/tmp.txt > $path/tmptmp.txt && mv $path/tmptmp.txt $path/tmp.txt

mv $path/tmp.txt $path/$out
