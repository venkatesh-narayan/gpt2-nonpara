#! /bin/bash
#
# make_trained_index.sh
# Copyright (C) 2022-02-10 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/make_trained_index.out
#SBATCH --error=slurm_output/make_trained_index.err
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=45g
#SBATCH --cpus-per-task=16
#SBATCH -t 0

python3 build_indices_on_ram.py \
    --txt_path wikitext_saves/dataset_train0.txt \
    --model_name_or_path gpt2-large \
    --idx 0 \
    --dstore_mmap wikitext/stride_512/dstore \
    --faiss_index wikitext/stride_512/knn.index \
    --starting_point 0 \
    --dstore_fp16
