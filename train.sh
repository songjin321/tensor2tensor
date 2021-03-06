#!/bin/bash

PROBLEM=headline_byte
MODEL=transformer
HPARAMS=transformer_headline
USER_DIR=./byteCup
STORAGE_BUCKET=gs://bytecup2018/tensor2tensor
DATA_DIR=$STORAGE_BUCKET/t2t_datagen
TRAIN_DIR=$STORAGE_BUCKET/t2t_train/headline_byte_GPU

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USER_DIR \
  --train_steps=500000 \
  --eval_steps=100 \
