# Decode
PROBLEM=headline_byte
MODEL=transformer
HPARAMS=transformer_headline
USER_DIR=./byteCup
STORAGE_BUCKET=gs://bytecup2018/tensor2tensor
DATA_DIR=$STORAGE_BUCKET/t2t_datagen
TRAIN_DIR=$STORAGE_BUCKET/t2t_train/headline_byte_GPU
OUTPUT_FILE=$STORAGE_BUCKET/t2t_train/headline.txt

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --eval_use_test_set=True \
  --decode_to_file=$OUTPUT_FILE \
  --t2t_usr_dir=$USER_DIR \