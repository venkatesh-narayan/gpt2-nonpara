#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022-03-21 Junxian <He>
#
# Distributed under terms of the MIT license.


import argparse
import torch
import faiss

import concurrent.futures
import copy
import math
import os
import collections

from tqdm import tqdm
from transformers import AutoTokenizer

import gc
import numpy as np

num_gpus = faiss.get_num_gpus()

def set_up_faiss(num_shards):
    indexes = []
    # reading all the indices at the same time causes script to be killed; instead,
    # we can just read the index here and then delete it after using it
    for shard_number in range(num_shards):
        index = faiss.read_index(f'webtext/stride_512/knn{shard_number}.index')
        print(f'\tdone reading index {shard_number}')

        # initialize resources and options
        # it did say in the faiss wiki that gpu indices are not thread safe because
        # it uses temp gpu memory or something -- could this be causing issues?
        # maybe can just set tempmem to 0 to avoid that?
        # see https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
        res = faiss.StandardGpuResources()
        res.setTempMemory(0)

        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.indicesOptions = faiss.INDICES_CPU
        print(f'\tmade resources and options for {shard_number}')

        # due to threading, i don't think this guarantees i'll search one thing on one gpu
        # but that might be ok
        device = shard_number % num_gpus
        index = faiss.index_cpu_to_gpu(res, device, index, co)

        print(f'\tput index {shard_number} on gpu {device}')

        indexes.append(index)

    return indexes


def search(index, query, shard_number, k):
    print('currently on: ', shard_number)

    dists, knns = index.search(query.detach().cpu().float().numpy(), k)
    print(f'\tdone searching shard number {shard_number}')

    # delete after finished using to save memory
    # del index
    # del res
    # del co
    # gc.collect()

    return torch.from_numpy(dists).cpu(), torch.from_numpy(knns).cpu()

def parallel_search_over_chunks(indexes, query, k, num_shards):
    import pdb; pdb.set_trace()
    # dists, knns = torch.empty((1, 1)), torch.empty((1, 1)) # initialize dists and knns to empty tensors
    dists, knns = None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        all_futures = [executor.submit(search, indexes[i], query, i, k) for i in range(num_shards)]

        for future in concurrent.futures.as_completed(all_futures):
            # import pdb; pdb.set_trace()
            i_dists, i_knns = future.result()
            print(i_dists)

            # in the beginning, there's nothing to cat; just want to set the variables
            # this is a better base case than enumerating over the futures and doing
            # something like "`if i == 0`" because for the first `max_workers` threads,
            # `i` will be 0, so there can be a chance for overwriting
            # if dists.shape == (1, 1) and knns.shape == (1, 1):
            #     dists, knns = copy.deepcopy(i_dists), copy.deepcopy(i_knns)
            #     print('\t\tSETTING INITIAL DISTS AND KNNS')
            # else:

            # to save memory, i can just find the top `k` dists at every instance
            # `dim = 1` because second dimension of the output of `index.search` corresponds to `k` nearest neighbors,
            # so want to add column-wise
            if dists is None:
                dists, knns = i_dists, i_knns
            else:
                dists, knns = torch.cat((dists, i_dists), dim=1), torch.cat((knns, i_knns), dim=1)

            # sort the `dists` and then get the top `k` of them; then, using the indices that we got from
            # sorting the `dists`, sort `knns` using that, and then take the top `k` of `knns`
            # here, `dim = 1` to sort by rows
            sorted_dists, indices = torch.sort(dists, dim=1, descending=False)
            dists = sorted_dists[:, :k]
            knns = torch.gather(knns, 1, indices) # sort knns row-wise by indices
            knns = knns[:, :k]

            print('\t\tSORTED DISTS AND TOOK TOP K')

        print(f'\t\t{dists.shape}, {knns.shape}')

        # not sure if this saves a lot of memory but doing it just in case
        del all_futures
        gc.collect()

    return dists, knns

seed = 0
if not os.path.exists(f'test_query{seed}.pt'):
    x = torch.randn(1536, 1280)
    torch.save(x, f'test_query{seed}.pt')
else:
    x = torch.load(f'test_query{seed}.pt')


# for debugging purpose
tok = AutoTokenizer.from_pretrained('gpt2-large')

keys = np.memmap('test_keys/dstore_keys.npy', dtype=np.float16, mode='r', shape=(62644, 1280))
vals = np.memmap('test_keys/dstore_vals.npy', dtype=np.int, mode='r', shape=(62644, 1))

x = torch.tensor(keys[:1536])

indexes = set_up_faiss(20)

dists, knns = parallel_search_over_chunks(indexes, x, 1024, 20)

import pdb; pdb.set_trace()

# below is for sanity check; i'm running the same query over the
# datastore with 10% of webtext (20 shards), so it should find
# the same neighbors (or at least mostly the same) as `knns`
print('now reading big index...')
big_index = faiss.read_index('webtext/stride_512/10_percent/knn_disk.index')

# because i'm searching the same vector every time i run this file, i can just do this search once,
# save them to a file, and then load them later
print('doing search on big index...')
if not os.path.exists(f'check_dists{seed}.npy'):
    check_dists, check_knns = big_index.search(x.float().numpy(), 1024)
    with open(f'check_dists{seed}.npy', 'wb') as f1:
        np.save(f1, check_dists)
    with open(f'check_knns{seed}.npy', 'wb') as f2:
        np.save(f2, check_knns)
else:
    with open(f'check_dists{seed}.npy', 'rb') as f1:
        check_dists = np.load(f2)
    with open(f'check_knns{seed}.npy', 'rb') as f2:
        check_dists = np.load(f2)

print('finished')
import pdb; pdb.set_trace()
print('finished pdb')


'''
below code was for playing around with the IndexShard stuff; commenting it out because we're not going to use it

def test_index_shards():
    index = faiss.read_index('webtext/stride_512/10_percent/knn_disk.index')


    def make_vres_vdev(i0, i1, tempmem):
        gpu_resources = []

        for i in range(num_gpus):
            res = faiss.StandardGpuResources()
            res.setTempMemory(int(tempmem))
            gpu_resources.append(res)

        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()

        for i in range(i0, i1):
            vres.push_back(gpu_resources[i])
            vdev.push_back(i)

        return vres, vdev, gpu_resources

    vres, vdev, gpu_resources = make_vres_vdev(args.i0, args.i1, args.tempmem)

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.verbose = True
    co.shard = True
    co.indicesOptions = faiss.INDICES_32_BIT # trying this out; default is INDICES_64_BIT. can also try INDICES_32_BIT

    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    x = torch.randn(1536, 1280) # same size as whats passed in hf

    dists, knns = index.search(x.cpu().float().numpy(), 1024)

    print(dists.shape, knns.shape)
'''
