# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Utility to convert saved_model to tflite."""
import sys

import kws_streaming.models.utils as utils
import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('saved_model_path', '',
                                 'Path to input saved_model.')
tf.compat.v1.flags.DEFINE_string('tflite_model_path', '',
                                 'Path to output tflite module.')

import numpy as np

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_train = np.float32(X_train)
# y_train = np.int32(y_train)
# Parameters
segment_length = 26
num_segments = 650 // segment_length

# Reshape X_train
# Reshape each row of X_train into 50 segments of length 13
X_train_reshaped = X_train.reshape(-1, num_segments, segment_length)

# Reshape y_train
# Replicate each label 50 times
y_train_replicated = np.repeat(y_train, num_segments)

# Reshape X_train_reshaped to the desired shape
X_train_final = X_train_reshaped.reshape(-1, segment_length)

print("X_train_final shape:", X_train_final.shape)  # Expected shape: (data_length * 50, 13)
print("y_train_replicated shape:", y_train_replicated.shape)  # Expected shape: (data_length * 50,)



X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_replicated))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def representative_data_gen():
  for input_value, _ in train_dataset.take(100):

    yield [input_value]

def main(unused_argv):
  del unused_argv  # Unused.

  with open(FLAGS.tflite_model_path, 'wb') as fd:
    fd.write(utils.saved_model_to_tflite(FLAGS.saved_model_path,
                                         optimizations=[tf.lite.Optimize.DEFAULT],
                                          representative_dataset=representative_data_gen
                                         ))


if __name__ == '__main__':
  FLAGS(sys.argv, known_only=True)
  FLAGS.unparse_flags()

  tf.compat.v1.app.run(main)
