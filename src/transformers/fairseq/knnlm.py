import torch
import torch.nn.functional as F
import faiss
import math
import numpy as np
import os
import concurrent.futures
import gc
#from fairseq import utils
import time
#from fairseq.data import Dictionary

class KNN_Dstore(object):
    def __init__(self, args, vocab_size, tokenizer=None):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

        self.use_gpu_faiss = args.use_gpu_faiss

        self.num_shards = args.num_shards

        self.shard_idxs_used = args.shard_idxs_used

        # adding support for multiple gpus
        # subtracting 1 to "reserve" cuda:0 for hf model
        self.num_parallelize = faiss.get_num_gpus() - 1 if args.use_gpu_faiss else os.cpu_count()
        if self.num_shards > 0: print(f'going to parallelize with {self.num_parallelize} threads')

        self.index = self.setup_faiss(args)


    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()

        # default to original usage if num_shards = 0
        if self.num_shards == 0:
            indexes = faiss.read_index(args.indexfile)#, faiss.IO_FLAG_ONDISK_SAME_DIR)

            #print('gpu faiss index')
            if self.use_gpu_faiss:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                indexes = faiss.index_cpu_to_gpu(res, 0, indexes, co)

            indexes.nprobe = args.probe

        else:
            indexes = []
            for shard_number in self.shard_idxs_used:
                # get current index name
                index_dir, index_name = os.path.split(args.indexfile)
                split_ext = os.path.splitext(index_name)
                index_name = split_ext[0] + str(shard_number) + split_ext[1]
                curr_indexfile = os.path.join(index_dir, index_name)

                if os.path.exists(curr_indexfile):
                    index = faiss.read_index(curr_indexfile)
                    print(f'\tdone reading index {shard_number}')

                    if self.use_gpu_faiss:
                        # initialize resources and options
                        # says in the faiss wiki that gpu indices are not thread safe because
                        # it uses temp gpu memory or something -- set tempmem to 0 to avoid this
                        # see https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
                        res = faiss.StandardGpuResources()
                        res.setTempMemory(0)

                        co = faiss.GpuClonerOptions()
                        co.useFloat16 = True
                        co.indicesOptions = faiss.INDICES_CPU
                        print(f'\tmade resources and options for {shard_number}')

                        # due to threading, i don't think this guarantees i'll search one query on one gpu
                        # but that might be ok
                        device = (shard_number % self.num_parallelize) + 1 # gives index in range [1, num_gpus]
                        index = faiss.index_cpu_to_gpu(res, device, index, co)

                        print(f'\tput index {shard_number} on gpu {device}')

                    index.nprobe = args.probe

                    indexes.append(index)
                else:
                    print(f'SKIPPING INDEX {shard_number}')

            print(f'using {len(indexes)} / {self.num_shards} shards')
        print('Reading datastores took {} s'.format(time.time() - start))

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return indexes


    def get_knns(self, index, queries):
        dists, knns = index.search(queries.detach().cpu().float().numpy(), self.k)

        if self.num_shards == 0:
            return dists, knns

        else:
            return torch.from_numpy(dists).cpu(), torch.from_numpy(knns).cpu()


    def parallel_search_over_shards(self, queries):
        dists, knns = None, None

        # set max_workers as num_gpus in order to limit the number of queries searched per gpu at a time
        # (prevent oom errors), but might be able to increase this number to get faster execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallelize) as executor:
            all_futures = [executor.submit(self.get_knns, self.index[i], queries) for i in range(len(self.index))]

            for future in concurrent.futures.as_completed(all_futures):
                i_dists, i_knns = future.result()

                if dists is None:
                    dists, knns = i_dists, i_knns
                else:
                    # `dim = 1` because second dimension of the output of `index.search` corresponds to `k` nearest neighbors,
                    # so want to add column-wise
                    dists, knns = torch.cat((dists, i_dists), dim=1), torch.cat((knns, i_knns), dim=1)

                # sort the `dists` and then get the top `k` of them; then, using the indices that we got from
                # sorting the `dists`, sort `knns` using that, and then take the top `k` of `knns`
                # so want to add column-wise
                sorted_dists, indices = torch.sort(dists, dim=1, descending=False)
                dists = sorted_dists[:, :self.k]
                knns = torch.gather(knns, 1, indices) # sort knns row-wise by indices
                knns = knns[:, :self.k]

                #print('\t\tSORTED DISTS AND TOOK TOP K')

            #print(f'\t\t{dists.shape}, {knns.shape}')

            # not sure if this saves a lot of memory but doing it just in case
            del all_futures
            gc.collect()

        return dists.numpy(), knns.numpy()


    def get_knn_log_prob(self, queries, src, pad_idx):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        #import pdb; pdb.set_trace()
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        src = src.contiguous().view(-1)

        # import pdb; pdb.set_trace()
        start_knn = time.time()

        if self.num_shards == 0:
            dists, knns = self.get_knns(self.index, queries[src != pad_idx])
        else:
            dists, knns = self.parallel_search_over_shards(queries[src != pad_idx])

        end_knn = time.time()
        # print(f'got dists and knns in {end_knn - start_knn} seconds')

        # (T_reducedxB)xK
        dists = queries.new_tensor(dists)
        start = time.time()
        dists = dist_func(dists, knns, queries[src != pad_idx, :], function=self.sim_func)
        probs = F.log_softmax(dists, dim=-1, dtype=torch.float32)

        # mid = time.time()
        # print(f'got probs in {mid - start} seconds')

        # index_mask = torch.eq(queries.new_tensor(self.vals[knns], dtype=torch.long).squeeze(-1), tgt[knn_mask].unsqueeze(-1)).float()
        # index_mask[index_mask == 0] = -10000 # for stability
        # index_mask[index_mask == 1] = 0

        #(TxB)xV
        # yhat_knn = torch.from_numpy(self.vals[knns])
        # return yhat_knn, probs
        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        indices = dists.new_tensor(self.vals[knns].squeeze(), dtype=torch.int64)

        # mapping: (T_reducedxB)xK
        unique_indices, mapping = torch.unique(indices, return_inverse=True)

        # (T_reducedxB)xKxn where n = num unique vals in indices
        knn_scores_by_index = dists.new_full([indices.size(0), indices.size(1), len(unique_indices)], -10000)
        knn_vals_by_index = dists.new_full([indices.size(0), indices.size(1), len(unique_indices)], 0, dtype=torch.long)

        # (T_reducedxB)xKx1
        indices = indices.unsqueeze(2)
        probs = probs.unsqueeze(2)
        mapping = mapping.unsqueeze(2)

        # (T_reducedxB)xKxn
        knn_scores_by_index.scatter_(dim=2, index=mapping, src=probs)
        knn_vals_by_index.scatter_(dim=2, index=mapping, src=indices)

        # (T_reducedxB)xKxn -> (T_reducedxB)xn
        # note that the last dimension is sparse with only one neuron activated
        knn_scores_by_index = knn_scores_by_index.logsumexp(dim=1)
        knn_vals_by_index = knn_vals_by_index.max(dim=1)[0]

        # import pdb; pdb.set_trace()
        # (T_reducedxB)xV
        full_knn_scores_hat = queries.new_full([knn_scores_by_index.shape[0], self.vocab_size], -10000)
        full_knn_scores_hat.scatter_(dim=1, index=knn_vals_by_index, src=knn_scores_by_index)

        # import pdb; pdb.set_trace()
        # (TxB)xV
        full_knn_scores = queries.new_full([queries.size(0), self.vocab_size], -10000)
        full_knn_scores[src != pad_idx] = full_knn_scores_hat

        return full_knn_scores.view(qshape[0], qshape[1], -1)

