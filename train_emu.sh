deepspeed --include localhost:0,1 src/train.py \
    --sample_weights balance \
    --goal_relabeling_strategy last_k_uniform \
    --goal_relabel_offset 10 \
    --goal_relabel_future_step 4 \
    --num_workers 2 \
    --dtype bf16 \
    --emu_ckpt /data1/hyx/huggingface/Emu/pretrain \
    --method emu \
    --train_batch_size 128 \
    --gradient_accumulation_steps 8 \
    --eval_batch_size 8 \
    --max_lr 1e-4 \
    --steps 100000 \
    --warmup_steps 3000 \
    --log_interval 1000 \
    --eval_interval 5000 \
    --save_interval 5000 \
    --save_dir emu_aug_save \
    --random_seed 42 \
    --zero_stage 2