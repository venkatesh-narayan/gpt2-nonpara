#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021-10-29 Junxian <He>
#
# Distributed under terms of the MIT license.

import os
import time
import math
import argparse

import torch
import numpy as np
import faiss
from itertools import chain

from datasets import load_dataset
import transformers
from transformers import (knnlmGPT2LMHeadModel,
                          GPT2Tokenizer,
                          AutoConfig,
                          Trainer,
                          TrainingArguments,
                          default_data_collator,
                          set_seed)
from transformers.testing_utils import CaptureLogger


parser = argparse.ArgumentParser()

parser.add_argument('--txt_path', type=str, help='path to read dataset from')
parser.add_argument('--model_name_or_path', type=str, help='name or path of model to use')
parser.add_argument('--chunk_size', type=int, default=int(1e4), help='size of each passage')
parser.add_argument('--stride', type=int, default=512, help='size of sliding window')
parser.add_argument('--max_keys_used', type=int, default=int(1e6), help='max number of keys used to train faiss index')
parser.add_argument('--idx', type=int, help='current shard index')

parser.add_argument('--lmbda', type=float, default=0.25, help='knn interpolation coefficient')
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, help='index to start adding keys at')

args = parser.parse_args()

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path,
                                          cache_dir=None,
                                          use_fast=True,
                                          revision="main",
                                          use_auth_token=None)
config = AutoConfig.from_pretrained(args.model_name_or_path,
                                    cache_dir=None,
                                    revision="main",
                                    use_auth_token=None)
args.dimension = config.n_embd
training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch", per_device_eval_batch_size=1)

def get_dstore_size(txt_path, idx):
    if not os.path.exists('dstore_sizes.txt'):
        f = open(txt_path, 'r')
        encodings = tokenizer('\n'.join([line for line in f]))
        f.close()

        f = open('dstore_sizes.txt', 'a')
        f.write(str(idx) + ' ' + str(len(encodings['input_ids'])) + '\n') # when parallelizing, need to know which index has which dstore size
        f.close()

        return len(encodings['input_ids'])
    else:
        f = open('dstore_sizes.txt', 'r')
        size = 0
        for line in f:
            tokens = line.split(' ')
            if int(tokens[0]) == idx:
                size = int(tokens[1])
                break

        f.close()

        return size

print('********** inferring datastore size (this could take some time for larger datasets) **********')
start_time = time.time()
args.dstore_size = get_dstore_size(args.txt_path, args.idx)
end_time = time.time()
print(f'took {end_time - start_time} s')

print(args)


def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples['text'])
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
        )
    return output

def prepare_shard(chunks):
    # apply group_text logic here
    max_length = config.n_positions
    stride = args.stride
    concatenated_chunks = { k: torch.Tensor(list(chain.from_iterable(v))) for k, v in chunks.items() } # flatten everything in given chunks

    total_length = concatenated_chunks['input_ids'].size(0)
    if total_length >= stride:
        total_length = (total_length // stride) * stride

    shard = { k: [] for k in concatenated_chunks.keys() }
    shard['labels'] = []
    for i in range(0, total_length, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, total_length)
        for k in concatenated_chunks.keys():
            # this happens in the beginning strides within the first max_length tokens
            if end_loc - begin_loc < max_length and begin_loc == 0:
                # padding
                to_append = torch.zeros(max_length)
                to_append[begin_loc:end_loc] = concatenated_chunks[k][begin_loc:end_loc]
            else:
                to_append = concatenated_chunks[k][begin_loc:end_loc]

            shard[k].append(to_append)

        labels = concatenated_chunks['input_ids'][begin_loc:end_loc].clone()
        trg_len = end_loc - i

        # do not compute loss for context
        labels[:-trg_len] = -100

        # this happens in the beginning strides within the first max_length tokens
        if end_loc - begin_loc < max_length and begin_loc == 0:
            # padding
            to_append = torch.zeros(max_length) - 100
            to_append[begin_loc:end_loc] = labels
        else:
            to_append = labels

        shard['labels'].append(to_append)

    # each value in the shard is a list of tensors -- stack them
    for k in shard.keys():
        shard[k] = torch.stack(shard[k]).long().numpy()
    return shard

def prepare_dataset(txt_path):
    raw_dataset = load_dataset('text', data_files={'train': txt_path}, keep_linebreaks=True)
    tokenized_dataset = raw_dataset.map(tokenize_function,
                                        batched=True,
                                        remove_columns=raw_dataset['train'].column_names,
                                        desc="running tokenizer on dataset")
    sharded_dataset = tokenized_dataset.map(prepare_shard,
                                            batched=True,
                                            remove_columns=tokenized_dataset['train'].column_names,
                                            desc="grouping chunks into shard")
    return sharded_dataset

def save_datastore_for_shard(sharded_dataset, curr_dstore_mmap):
    # add relevant knnlm args to config to save datastore
    config.stride            = args.stride
    config.save_knnlm_dstore = True
    config.knnlm             = False
    config.dstore_mmap       = curr_dstore_mmap
    config.dstore_size       = args.dstore_size
    config.faiss_index       = None
    config.lmbda             = args.lmbda

    # set seed before initializing model
    set_seed(training_args.seed)

    model = knnlmGPT2LMHeadModel.from_pretrained(args.model_name_or_path,
                                                 from_tf=bool(".ckpt" in args.model_name_or_path),
                                                 config=config,
                                                 cache_dir=None,
                                                 revision="main",
                                                 use_auth_token=None)
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=None,
                      eval_dataset=sharded_dataset['train'],
                      tokenizer=tokenizer,
                      data_collator=default_data_collator)

    print('********** EVALUATE **********')
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(sharded_dataset['train'])
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    #trainer.save_metrics("eval", metrics)


def build_faiss_index(sharded_dataset, flag):
    print(f'SANITY CHECK: TXT PATH IS {args.txt_path} AND IDX IS {args.idx}')
    idx = args.idx

    split_path = os.path.split(args.dstore_mmap)
    path_to_check = os.path.join(split_path[0], str(idx))
    if not os.path.exists(path_to_check):
        os.makedirs(path_to_check, exist_ok=True)

    curr_dstore_mmap = os.path.join(path_to_check, split_path[1])
    if flag: save_datastore_for_shard(sharded_dataset, curr_dstore_mmap)

    if args.dstore_fp16:
        keys = np.memmap(curr_dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
        vals = np.memmap(curr_dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
    else:
        keys = np.memmap(curr_dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
        vals = np.memmap(curr_dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

    if not os.path.exists(args.faiss_index + '.trained'):
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension, args.ncentroids, args.code_size, 8)
        index.nprobe = args.probe

        print('Training Index')
        np.random.seed(args.seed)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(args.max_keys_used, vals.shape[0])], replace=False)
        start = time.time()
        index.train(keys[random_sample].astype(np.float32))
        print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()
        faiss.write_index(index, args.faiss_index + '.trained')
        print('Writing index took {} s'.format(time.time() - start))

    print('Adding Keys')
    index = faiss.read_index(args.faiss_index + '.trained')
    start = args.starting_point
    start_time = time.time()

    split_ext = os.path.splitext(args.faiss_index)
    separate_index_name = split_ext[0] + str(idx) + split_ext[1]

    while start < args.dstore_size:
        end = min(args.dstore_size, start + args.num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += args.num_keys_to_add_at_a_time

        if start % 1000000 == 0:
            print('Added %d tokens so far' % start)
            print('Writing Index', start)

            faiss.write_index(index, separate_index_name)

    print('Adding total %d keys' % start)
    print('Adding took {} s'.format(time.time() - start_time))
    print('Writing Index')
    start = time.time()
    faiss.write_index(index, separate_index_name)
    print('Writing index took {} s'.format(time.time() - start))
    print()

    os.remove(curr_dstore_mmap + '_keys.npy') # remove keys after adding to faiss index


#if __name__ == '__main__':
    #sharded_dataset = prepare_dataset(args.txt_path)
    #build_faiss_index(sharded_dataset)

