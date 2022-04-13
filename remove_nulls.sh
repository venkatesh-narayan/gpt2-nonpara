#! /bin/bash
#
# remove_nulls.sh
# Copyright (C) 2022-03-04 Junxian <He>
#
# Distributed under terms of the MIT license.
#


for filename in ./webtext_saves/*.txt; do
    out=$(basename "$filename")
    out=fixed_$out
    tr -d '\000' < $filename > ./webtext_saves/$out
done
