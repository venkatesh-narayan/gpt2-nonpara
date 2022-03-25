#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022-01-31 Junxian <He>
#
# Distributed under terms of the MIT license.

import os
import time
import argparse

import transformers
from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, help='path to read dataset from. DO NOT INCLUDE FILE NAMES!')
parser.add_argument('--model_name_or_path', type=str, help='name or path of model to use')

args = parser.parse_args()

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path,
                                          cache_dir=None,
                                          use_fast=True,
                                          revision="main",
                                          use_auth_token=None)

def get_dstore_sizes(path):
    fnames = [p for p in os.listdir(path) if '.txt' in p] # include only text files from path
    fnames = [os.path.join(path, p) for p in fnames]
    fnames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    f = open('dstore_sizes.txt', 'w')
    for idx, fname in enumerate(fnames):
        curr_file = open(fname, 'r')
        encodings = tokenizer('\n'.join([line for line in curr_file]))
        curr_file.close()

        f.write(str(idx) + ' ' + str(len(encodings['input_ids'])) + '\n')
    f.close()

if __name__ == "__main__":
    get_dstore_sizes(args.path)






