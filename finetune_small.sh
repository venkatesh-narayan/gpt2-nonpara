#! /bin/bash
#
# finetune_small.sh
# Copyright (C) 2021-11-02 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/finetune_small.out
#SBATCH --error=slurm_output/finetune_small.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=20
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH --nodelist=compute-0-31,compute-0-30
#SBATCH -t 0

export WANDB_PROJECT=gpt2_lm
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`

model="gpt2"

max_steps=100000
lr=5e-5
lr_scheduler_type="polynomial"
warmup_updates=0
bsz=16
gradient_steps=4

SAVE=

python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --stride 1024 \
    --do_train \
    --do_eval \
    --output_dir checkpoints/gpt2/
