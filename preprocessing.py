#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021-10-29 Junxian <He>
#
# Distributed under terms of the MIT license.

import os
import time
import argparse
import random
import subprocess

import torch
import numpy as np

from datasets import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, nargs='?', help='name of dataset')
parser.add_argument('--dataset_config', type=str, nargs='?', help='config of dataset')
parser.add_argument('--in_path', type=str, nargs='?', help='path of dataset txt file')
parser.add_argument('--out_path', type=str, help='path to write/read dataset from')
parser.add_argument('--num_shards', type=int, default=20, help='number of shards to use')

args = parser.parse_args()
print(args)

def chunk_and_shuffle(text, chunk_size, delimiter):
    merged_text = f'{delimiter}'.join(text) # get full text
    chunked = [merged_text[i:i+chunk_size] for i in range(0, len(merged_text), chunk_size)] # chunk into "paragraphs"
    random.shuffle(chunked) # permute list
    return '\n'.join(chunked)

def dataset_to_txt(out_path, dataset_name, config, num_shards):
    if num_shards <= 0:
        raise ValueError('need a positive number of shards')

    check_existence = os.path.splitext(out_path)
    check_existence = check_existence[0] + '0' + check_existence[1]
    if not os.path.exists(check_existence):
        os.makedirs(os.path.split(out_path)[0], exist_ok=True) # make directories in case they don't exist

        dataset = load_dataset(dataset_name, config, split='train')

        print('********** WRITING **********')
        size_of_shard = len(dataset['text']) // num_shards
        splits = [dataset['text'][size_of_shard * i:size_of_shard * (i + 1)] for i in range(num_shards)]

        start = time.time()
        for i, arr in enumerate(splits):
            curr_path = os.path.splitext(out_path)
            curr_path = curr_path[0] + str(i) + curr_path[1]
            f = open(curr_path, 'w')

            shuffled_text = chunk_and_shuffle(arr, size_of_shard, '\n')
            f.write(shuffled_text + '\n')
            f.close()
        end = time.time()
        print('********** FINISHED WRITING **********')
        print('took {end - start} seconds to write files')
    else:
        print(f'already wrote dataset to txt file(s)')

# instead of loading from huggingface datasets, use text file that has the dataset in it
def split_file(in_path, out_path, num_shards):
    if num_shards <= 0:
        raise ValueError('need a positive number of shards')

    check_existence = os.path.splitext(out_path)
    check_existence = check_existence[0] + '0' + check_existence[1]
    if not os.path.exists(check_existence):
        os.makedirs(os.path.split(out_path)[0], exist_ok=True) # make directories in case they don't exist

        num_words = subprocess.check_output(['wc', '-w', in_path], stderr=subprocess.STDOUT) # this is in bytes
        num_words = str(num_words).split("'")[1] # the first one will always be "b"
        num_words = int(num_words.split(' ')[0])
        print("NUM_WORDS = ", num_words)
        size_of_shard = num_words // num_shards
        print("size of shard = ", size_of_shard)

        f = open(in_path, 'r')
        i = 0
        curr_split = [] # matching format of splits so that i can still use chunk_and_shuffle
        for line in f:
            #print(len(splits), len(curr_split))
            tokens = line.split(' ')
            if len(curr_split) + len(tokens) <= size_of_shard:
                curr_split.extend(tokens)
            else:
                to_add = size_of_shard - len(curr_split)
                curr_split.extend(tokens[:to_add])

                shuffled_text = chunk_and_shuffle(curr_split, size_of_shard, ' ')

                curr_path = os.path.splitext(out_path)
                curr_path = curr_path[0] + str(i) + curr_path[1]
                nf = open(curr_path, 'w')
                nf.write(shuffled_text + '\n')
                nf.close()

                print(f'wrote to {curr_path}, has {len(curr_split)} words')

                i += 1

                curr_split = tokens[to_add:]

        if len(curr_split) > 0:
            shuffled_text = chunk_and_shuffle(curr_split, size_of_shard, ' ')
            curr_path = os.path.splitext(out_path)
            curr_path = curr_path[0] + str(i) + curr_path[1]
            nf = open(curr_path, 'w')
            nf.write(shuffled_text + '\n')
            nf.close()

            print(f'wrote to {curr_path}, has {len(curr_split)} words')

        f.close()

if __name__ == '__main__':
    if args.dataset_name is not None and args.dataset_config is not None and args.in_path is not None:
        raise ValueError('only specify either (dataset_name and dataset_config) or (in_path)')
    elif args.dataset_name is not None and args.dataset_config is not None:
        dataset_to_txt(args.out_path, args.dataset_name, args.dataset_config, args.num_shards)
    elif args.in_path is not None:
        split_file(args.in_path, args.out_path, args.num_shards)
    else:
        raise ValueError('must specify at least one method of creating dataset files')
