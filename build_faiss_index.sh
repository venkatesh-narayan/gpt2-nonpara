#! /bin/bash
#
# build_faiss_index.sh
# Copyright (C) 2021-10-21 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --mem=20g
#SBATCH --cpus-per-task=10
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30
#SBATCH -t 0

python build_dstore.py \
    --dstore_mmap checkpoints/dstore \
    --dstore_size 119721489 \
    --faiss_index checkpoints/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 \
    --dstore_fp16
