# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import tarfile
import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import *
import json
import tensorflow as tf

BUCKET = 'bytecup2018' 
assert BUCKET, 'Must specify an existing GCS bucket name'
TASK_DATA_DIR = 'gs://{}/bytecup2018'.format(BUCKET)
train_files_size = 8
CONTENT_MAX_LENGTH = 412
TITLE_MAX_LENGTH = 100

@registry.register_problem
class HeadlineByte(text_problems.Text2TextProblem):
  """Headline generation for byte competetion"""
  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    },{
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    """Generate samples."""
    if dataset_split == problem.DatasetSplit.TRAIN:
        for i in range(train_files_size):
            TRAIN_DATA_PATH = 'gs://{}/bytecup2018/bytecup.corpus.train.{}.txt'.format(BUCKET,i)
            with tf.gfile.Open(TRAIN_DATA_PATH, "r") as f:
                lines = f.readlines()
                for line in lines:
                    story = json.loads(line)['content'][:CONTENT_MAX_LENGTH]
                    summary = json.loads(line)['title'][:TITLE_MAX_LENGTH]
                    yield {"inputs": story, "targets": summary}
    elif dataset_split == problem.DatasetSplit.EVAL:
        EVAL_DATA_PATH = os.path.join(TASK_DATA_DIR, "bytecup.corpus.train.8.txt")
        with tf.gfile.Open(EVAL_DATA_PATH, "r") as f:
            lines = f.readlines()
            for line in lines[:int(len(lines)/2)]:
                story = json.loads(line)['content'][:CONTENT_MAX_LENGTH]
                summary = json.loads(line)['title'][:TITLE_MAX_LENGTH]
                yield {"inputs": story, "targets": summary}
    else:
        TEST_DATA_PATH = os.path.join(TASK_DATA_DIR, "bytecup.corpus.train.8.txt")
        with tf.gfile.Open(TEST_DATA_PATH, "r") as f:
            lines = f.readlines()
            for line in lines[int(len(lines)/2):]:
                story = json.loads(line)['content'][:CONTENT_MAX_LENGTH]
                summary = json.loads(line)['title'][:TITLE_MAX_LENGTH]
                yield {"inputs": story, "targets": summary}

  @property
  def vocab_filename(self):
    return wiki_lm.LanguagemodelEnWiki32k().vocab_filename
  @property
  def packed_length(self):
    return 512

@registry.register_problem
class HeadlineTest(HeadlineByte):
  """Headline Test for byte competetion"""
  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def vocab_filename(self):
    return HeadlineByte().vocab_filename

  @property
  def packed_length(self):
    return None

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    """Generate samples."""
    if dataset_split == problem.DatasetSplit.TEST:
        EVAL_DATA_PATH = os.path.join(TASK_DATA_DIR, "bytecup.corpus.validation_set.txt")
        with tf.gfile.Open(EVAL_DATA_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                story = json.loads(line)['content'][:CONTENT_MAX_LENGTH]
                yield {"inputs": story, "targets": ''}

@registry.register_hparams
def transformer_headline():
  hparams = transformer_big()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  update_hparams_for_tpu(hparams)
  return hparams

@registry.register_hparams
def transformer_headline_test():
  hparams = transformer_base()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4096
  return hparams