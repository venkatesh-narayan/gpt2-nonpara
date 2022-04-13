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
parser.add_argument('--dstore_mmap', type=str, help='location of datastore')
parser.add_argument('--dstore_out_path', type=str, help='location to write dstore files')

args = parser.parse_args()
print(args)

'''
don't need this anymore
def merge_all(faiss_index, num_shards, out_path):
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
    merge_ondisk(index, index_names, out_path + '.ivfdata')
    end = time.time()
    print('********** FINISHED MERGING ON DISK **********')
    print(f'took {end - start} s')

    print('********** WRITING INDEX **********')
    start = time.time()

    split_ext = os.path.splitext(out_path)
    faiss.write_index(index, split_ext[0] + '_disk' + split_ext[1])
    end = time.time()
    print('********** FINISHED WRITING INDEX **********')
    print(f'took {end - start} s')
'''

def concatenate_vals(dstore_mmap, num_shards, out_path):
    f = open('dstore_sizes.txt', 'r')
    dstore_sizes = []
    for i, line in enumerate(f):
        if i == num_shards:
            break

        tokens = line.split(' ')
        dstore_sizes.append(int(tokens[1]))
    f.close()

    dstore_size = sum(dstore_sizes)
    vals = np.memmap(out_path+'_vals.npy', dtype=np.int, mode='w+', shape=(dstore_size, 1))
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
    #merge_all(args.faiss_index, args.num_shards, args.faiss_out_path)
    concatenate_vals(args.dstore_mmap, args.num_shards, args.dstore_out_path)
