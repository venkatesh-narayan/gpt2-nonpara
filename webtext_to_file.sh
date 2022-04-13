#! /bin/bash
#
# webtext_to_file.sh
# Copyright (C) 2022-02-22 Junxian <He>
#
# Distributed under terms of the MIT license.
#


for filename in ./openwebtext/*.xz; do
    unxz < $filename >> ./openwebtext/all.txt
    echo "\n" >> ./openwebtext/all.txt
    rm $filename
done
