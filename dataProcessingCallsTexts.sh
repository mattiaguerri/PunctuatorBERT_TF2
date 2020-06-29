#!/bin/bash

path=./DataCallsTexts
inp=CALLS_TEXTS_00.txt
out=CALLS_TEXTS_01.txt

# Remove blank lines, convert upper case to lower case.
grep "\S" $path/$inp | tr '[:upper:]' '[:lower:]' > $path/tmp_00

# Remove numeric characters
sed 's/[0-9]*//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove <> and anything in between.
# sed 's/<[^~]*>//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00
sed 's/<[^>]*>//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace space+dash with dash.
sed 's/ -/-/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace dash+space with space.
sed 's/- / /g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Replace apostrphe+space with apostrophe.
sed s/"'"\ /"'"/g $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# Remove the pipe.
sed 's/|//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00




# # Replace group of three dots ("...") with one dot.
# sed 's/\.\.\././g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Remove the pattern double quotes + space ("" ").
# sed 's/ "//g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Remove the pattern space + double quotes (" "").
# sed 's/" //g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Replace space + colon with colon.
# sed 's/ :/:/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Replace space + exclamation mark with exclamation mark.
# sed 's/ !/!/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Replace space + question mark with question mark.
# sed 's/ ?/?/g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Remove pattern dash + space ("- ")
# sed 's/- //g' $path/tmp_00 > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

# # Replace comma, semicolon, colon, exclamation mark and question mark with period.
# sed 's/,/./g' $path/tmp_00 | sed 's/;/./g' | sed 's/:/./g' | sed 's/!/./g' | sed 's/?/./g' > $path/tmp_01 && mv $path/tmp_01 $path/tmp_00

mv $path/tmp_00 $path/$out
