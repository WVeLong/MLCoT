#!/bin/bash

echo $PWD

deepspeed --include localhost:0,1,2,3\
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --training_stage 'pretrain' \
    --use_MLCoT False \
    --MLCoT_num 10 \
    --model_name_or_path /ssd/common/LLMs/vicuna-7b-v1.5/ \
    --version v1 \
    --data_path /ssd/common/datasets/medical-image-analysis/LLaVA-Pretrain-new/blip_laion_cc_sbu_558k.json \
    --image_folder /ssd/common/datasets/medical-image-analysis/LLaVA-Pretrain-new/image/ \
    --vision_tower /hdd/wuwl/model/clip-vit-large-patch14/ \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/viscot-7b-224-pretrain-baseline-new \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --ft_vision_tower True