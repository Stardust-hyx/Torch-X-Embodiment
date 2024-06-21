deepspeed --include localhost:0,1,2,3,4,5 ./src/train.py \
    --data_dir $HYXDIR/np_datasets \
    --sample_weights balance \
    --use_history True \
    --num_workers 8 \
    --prompt_type rt1_default \
    --dtype bf16 \
    --text_enc $HYXDIR/huggingface/bge-base-en-v1.5 \
    --method rt1 \
    --train_batch_size 216 \
    --gradient_accumulation_steps 3 \
    --eval_batch_size 96 \
    --max_lr 5e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.001 \
    --steps 100000 \
    --warmup_steps 10000 \
    --warmup_type linear \
    --log_interval 2000 \
    --eval_interval 4000 \
    --save_interval 4000 \
    --start_save 80000 \
    --save_dir $HYXDIR/save_rt1/rt1_pretrain \
    --random_seed 41