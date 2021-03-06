#!/bin/bash

STORAGE_BUCKET=gs://bytecup2018/tensor2tensor
PROBLEM=headline_byte
DATA_DIR=$STORAGE_BUCKET/t2t_datagen/
USER_DIR=./byteCup

# Generate data
t2t-datagen \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR \
  --t2t_usr_dir=$USER_DIR \