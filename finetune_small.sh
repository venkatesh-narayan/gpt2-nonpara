#! /bin/bash
#
# finetune_small.sh
# Copyright (C) 2021-11-02 Junxian <He>
#
# Distributed under terms of the MIT license.
#


#SBATCH --output=slurm_output/slurm-%A-%a.out
#SBATCH --error=slurm_output/slurm-%A-%a.out
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --nodes=1
#SBATCH --job-name=gpt2lm
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH --nodelist=compute-0-31,compute-0-30

export WANDB_PROJECT=gpt2_lm
export WANDB_WATCH="false"
export WANDB_ENTITY="gpt2-nonpara"

DATE=`date +%Y%m%d`

model="gpt2-large"
dataset="wikitext"
report_to="wandb"

max_steps=100000
port=62227

# learning hyperparams
lr=5e-5
max_grad_norm=1
lr_scheduler_type="polynomial"
warmup_steps=5000
weight_decay=0

bsz=2
gradient_steps=4

logging_steps=100

eval_strategy="steps"
# eval_strategy="steps"
save_steps=3000


debug=0
extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    max_train_samples=4000
    max_eval_samples=150
    bsz=10
    gradient_steps=1
    num_train_epochs=30
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_eval_samples ${max_eval_samples}"
    debug_str=".debug"
fi


exp_name=${model}.${dataset}.warm${warmup_steps}.wd${weight_decay}${debug_str}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}

# python -u examples/pytorch/language-modeling/run_clm.py \
python -m torch.distributed.launch --nproc_per_node 2 --master_port=${port} examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path ${model} \
    --dataset_name ${dataset} \
    --dataset_config_name wikitext-103-raw-v1 \
    --stride 1024 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_steps} \
    --report_to ${report_to} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --fp16 \
    --run_name ${DATE}.${exp_name} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --overwrite_output_dir "True" \
    --disable_tqdm "True" \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
