#! /bin/bash
#
# make_trained_index.sh
# Copyright (C) 2022-02-10 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/make_trained_index.out
#SBATCH --error=slurm_output/make_trained_index.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=15g
#SBATCH -t 0

python3 build_indices_on_ram.py \
    --txt_path wikitext_saves/dataset_train0.txt \
    --model_name_or_path gpt2 \
    --idx 0 \
    --dstore_mmap test_preprocessing/stride_512/dstore \
    --faiss_index test_preprocessing/stride_512/knn.index \
    --starting_point 0 \
    --dstore_fp16
