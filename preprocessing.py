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

import torch
import numpy as np

from datasets import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='openwebtext', help='name of dataset')
parser.add_argument('--dataset_config', type=str, default='plain_text', help='config of dataset')
parser.add_argument('--out_path', type=str, help='path to write/read webtext dataset from')
parser.add_argument('--num_shards', type=int, default=20, help='number of shards to use')

args = parser.parse_args()
print(args)

def chunk_and_shuffle(text, chunk_size):
    merged_text = '\n'.join(text) # get full text
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

            shuffled_text = chunk_and_shuffle(arr, size_of_shard)
            f.write(shuffled_text + '\n')
            f.close()
        end = time.time()
        print('********** FINISHED WRITING **********')
        print(f'took {end - start} seconds to write to file {out_path}')
    else:
        print(f'already wrote dataset to txt file(s)')



if __name__ == '__main__':
    dataset_to_txt(args.out_path, args.dataset_name, args.dataset_config, args.num_shards)
