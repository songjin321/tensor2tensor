#!/bin/bash

PROBLEM=headline_byte
MODEL=transformer
HPARAMS=transformer_prepend

STORAGE_BUCKET=gs://bytecup2018/tensor2tensor
DATA_DIR=$STORAGE_BUCKET/t2t_data/t2t_data
TRAIN_DIR=$STORAGE_BUCKET/t2t_train/t2t_train/headline_byte/transformer-transformer_prepend
USER_DIR=$HOME/tensor2tensor/byteCup

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USER_DIR \
  --train_steps=50000 \
  --eval_steps=1000 \
  --use_tpu \
  --cloud_tpu_name=$TPU_NAME

