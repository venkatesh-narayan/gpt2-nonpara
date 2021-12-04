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

import numpy as np

import faiss
from faiss.contrib.ondisk import merge_ondisk


parser = argparse.ArgumentParser()

parser.add_argument('--num_shards', type=int, default=20, help='number of shards created')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--dstore_mmap', type=str, help='location of datastore')

args = parser.parse_args()
print(args)

def merge_all(faiss_index, num_shards):
    # by this point, all the indices should have been written
    print('********** LOADING TRAINED INDEX **********')
    start = time.time()
    index = faiss.read_index(faiss_index + '.trained')
    end = time.time()
    print('********** FINISHED LOADING TRAINED INDEX **********')
    print(f'took {end - start} s')

    split_ext = os.path.splitext(faiss_index)
    index_names = [split_ext[0] + str(idx) + split_ext[1] for idx in range(num_shards)]
    index_names = [name for name in index_names if os.path.exists(name)]

    print('********** MERGING ON DISK **********')
    start = time.time()
    merge_ondisk(index, index_names, faiss_index + '.ivfdata')
    end = time.time()
    print('********** FINISHED MERGING ON DISK **********')
    print(f'took {end - start} s')

    print('********** WRITING INDEX **********')
    start = time.time()
    faiss.write_index(index, split_ext[0] + '_disk' + split_ext[1])
    end = time.time()
    print('********** FINISHED WRITING INDEX **********')
    print(f'took {end - start} s')


def concatenate_vals(dstore_mmap, num_shards):
    f = open('dstore_sizes.txt', 'r')
    dstore_sizes = dict()
    for line in f:
        tokens = line.split(' ')
        dstore_sizes[int(tokens[0])] = int(tokens[1])
    f.close()

    dstore_size = sum(dstore_sizes.values())
    vals = np.memmap(dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(dstore_size, 1))
    curr_position = 0
    for idx in range(num_shards):
        split_path = os.path.split(dstore_mmap)
        curr_dstore_mmap = os.path.join(split_path[0], str(idx), split_path[1])
        if os.path.exists(curr_dstore_mmap+'_vals.npy'):
            intermediate_vals = np.memmap(curr_dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(dstore_sizes[idx], 1))
            vals[curr_position:curr_position + dstore_sizes[idx], :] = intermediate_vals
            curr_position += dstore_sizes[idx]

            #os.remove(curr_dstore_mmap+'_vals.npy') # remove intermediate vals once finished concatenating


if __name__ == '__main__':
    #merge_all(args.faiss_index, args.num_shards)
    concatenate_vals(args.dstore_mmap, args.num_shards)
