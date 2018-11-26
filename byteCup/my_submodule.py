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
from tensor2tensor.model.transformer import *
import json
import tensorflow as tf

# Links to data from http://cs.nyu.edu/~kcho/DMQA/
BUCKET = 'bytecup2018' #@param {type:"string"}
assert BUCKET, 'Must specify an existing GCS bucket name'
TASK_DATA_DIR = 'gs://{}/bytecup2018'.format(BUCKET)
TRAIN_DATA_PATH = os.path.join(TASK_DATA_DIR, "bytecup.corpus.train.0.txt")
print('***** Task data directory: {} *****'.format(TASK_DATA_DIR))

def _story_summary_split(story):
  split_str = u" <summary> "
  split_str_len = len(split_str)
  split_pos = story.find(split_str)
  return story[:split_pos], story[split_pos + split_str_len:]  # story, summary

def example_generator(all_files, sum_token):
    """Generate examples."""
    with tf.gfile.Open(all_files, "r") as f:
        lines = f.readlines()
    for (i, line) in enumerate(lines):
        story = json.loads(line)['content']
        summary = json.loads(line)['title']
        story_summary_split_token = u" <summary> " if sum_token else " "
        yield " ".join(story) + story_summary_split_token + " ".join(summary)

@registry.register_problem
class HeadlineByte(text_problems.Text2TextProblem):
  """Headline generation for byte competetion"""

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    del data_dir
    all_files = TRAIN_DATA_PATH
    return example_generator(all_files, sum_token=False)

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1000,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def is_generate_per_split(self):
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    all_files = TRAIN_DATA_PATH
    for example in example_generator(all_files, sum_token=True):
      story, summary = _story_summary_split(example)
      yield {"inputs": story, "targets": summary}

@registry.register_hparams
def transformer_headline():
  hparams = transformer_base_v2()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 256
  hparams.batch_size = 64
  update_hparams_for_tpu(hparams)
  return hparams
