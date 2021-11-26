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

import faiss
from faiss.contrib.ondisk import merge_ondisk


parser = argparse.ArgumentParser()

parser.add_argument('--num_shards', type=int, default=20, help='number of shards created')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')

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


if __name__ == '__main__':
    merge_all(args.faiss_index, args.num_shards)

