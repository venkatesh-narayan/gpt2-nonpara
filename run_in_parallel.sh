#! /bin/bash
#
# run_in_parallel.sh
# Copyright (C) 2021-11-20 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/run_in_parallel-%A_%a.out
#SBATCH --error=slurm_output/run_in_parallel-%A_%a.err
#SBATCH --array=0-19%7
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=30g
#SBATCH -t 0

taskid=${SLURM_ARRAY_TASK_ID}

python3 build_indices_on_ram.py \
    --txt_path wikitext_saves/dataset_train${taskid}.txt \
    --model_name_or_path gpt2 \
    --idx ${taskid} \
    --dstore_mmap test_preprocessing/stride_512/dstore \
    --faiss_index test_preprocessing/stride_512/knn.index \
    --starting_point 0 \
    --dstore_fp16
