#! /bin/bash
#
# finetune_medium.sh
# Copyright (C) 2021-11-02 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/finetune_medium.out
#SBATCH --error=slurm_output/finetune_medium.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=20
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH --nodelist=compute-0-31,compute-0-30
#SBATCH -t 0

python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2-medium \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --stride 1024 \
    --do_train \
    --do_eval \
    --output_dir checkpoints/gpt2_medium/
