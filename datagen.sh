#!/bin/bash

PROBLEM=headline_byte
MODEL=transformer
HPARAMS=transformer_prepend

STORAGE_BUCKET=gs://bytecup2018/tensor2tensor
DATA_DIR=$STORAGE_BUCKET/t2t_data/
TMP_DIR=$HOME/t2t_tmp
USER_DIR=/content/tensor2tensor/byteCup

mkdir -p $TMP_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$USER_DIR