#! /bin/bash
#
# eval_knnlm_small.sh
# Copyright (C) 2021-10-22 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/eval_nnlm_small.out
#SBATCH --error=slurm_output/eval_nnlm_small.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=20
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30
#SBATCH -t 0

python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --do_eval \
    --stride 1024 \
    --is_knnlm_model \
    --knnlm \
    --per_device_eval_batch_size 24 \
    --output_dir ./knnlmgpt2_1024 \
    --report_to none
