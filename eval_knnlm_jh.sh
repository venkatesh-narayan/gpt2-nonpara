



python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2 --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 --stride 512 \
    --is_knnlm_model --knnlm --dstore_mmap test_preprocessing/stride_512/10/dstore --dstore_size 5947761 \
    --faiss_index test_preprocessing/stride_512/knn10.index \
    --do_eval --per_device_eval_batch_size 1 \
    --output_dir ./tmp --report_to none