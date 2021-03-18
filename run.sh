#!/bin/bash

python main.py \
--optimizer adamw \
--batch_size 32 \
--global_lr 1e-03 \
--average_mha 1 \
--num_gat_layers 6 \
--graph_conv_in_dim 64 \
--graph_conv_out_dim 64 \
--gat_conv_num_heads 4 \
--lr_scheduler reduce_on_plateau \
--reduce_on_plateau_lr_scheduler_patience 10 \
--reduce_on_plateau_lr_scheduler_threshold 0.02 \
--use_pe 1 \
--use_prune 1 \
--use_ffn 0 \
--remove_isolated 0 \
--prune_keep_p 0.8 \
--loss_type sl1 \
--use_conv1d 0 \
--use_loss_norm 1 \
--use_all_to_all 0 \
--epoch_num 15 \
--useGNN 1 \
--task mosi_unaligned \
--dataroot <dataroot> \
--log_dir debug

