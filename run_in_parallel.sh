#! /bin/bash
#
# run_in_parallel.sh
# Copyright (C) 2021-11-20 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/run_in_parallel-%A_%a.out
#SBATCH --error=slurm_output/run_in_parallel-%A_%a.err
#SBATCH --array=40-80%7
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=45g
#SBATCH --cpus-per-task=32
#SBATCH -t 0

taskid=${SLURM_ARRAY_TASK_ID}

python3 build_indices_on_ram.py \
    --txt_path fixed_webtext_saves/fixed_dataset_train${taskid}.txt \
    --model_name_or_path gpt2-large \
    --idx ${taskid} \
    --dstore_mmap webtext/stride_512/dstore \
    --faiss_index webtext/stride_512/knn.index \
    --starting_point 0 \
    --dstore_fp16
