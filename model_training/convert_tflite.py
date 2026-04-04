# coding=utf-8
# Copyright 2024 The Google Research Authors.
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


import tensorflow as tf
import kws_streaming.layers.quantize as quantize
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1

def saved_model_to_tflite(saved_model_path,
                          optimizations=None,
                          inference_type=tf1.lite.constants.INT8,
                          experimental_new_quantizer=True,
                          representative_dataset=None,
                          inference_input_type=tf.float32,
                          inference_output_type=tf.float32,
                          use_quantize_nbit=0):
  """Convert saved_model to tflite.

  Args:
    saved_model_path: path to saved_model
    optimizations: list of optimization options
    inference_type: inference type, can be float or int8
    experimental_new_quantizer: enable new quantizer
    representative_dataset: function generating representative data sets
      for calibation post training quantizer
    inference_input_type: it can be used to quantize input data e.g. tf.int8
    inference_output_type: it can be used to quantize output data e.g. tf.int8
    use_quantize_nbit: adds experimental flag for default_n_bit precision.

  Returns:
    tflite model
  """

  # Identify custom objects.
  with quantize.quantize_scope():
    print("ok")
    converter = tf.compat.v2.lite.TFLiteConverter.from_saved_model(
        saved_model_path)
  converter.experimental_enable_resource_variables = True
  converter.inference_type = inference_type
  converter.experimental_new_quantizer = experimental_new_quantizer
  converter.experimental_enable_resource_variables = True
  converter.experimental_new_converter = True
  if representative_dataset is not None:
    converter.representative_dataset = representative_dataset

  # this will enable audio_spectrogram and mfcc in TFLite
#   converter.target_spec.supported_ops = [
#       tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
#   ]
#   converter.allow_custom_ops = True

  converter.inference_input_type = inference_input_type
  converter.inference_output_type = inference_output_type
  if optimizations:
    converter.optimizations = optimizations
#   if use_quantize_nbit:
#     # pylint: disable=protected-access
#     converter._experimental_low_bit_qat = True
    # pylint: enable=protected-access
  print("before convert")
  tflite_model = converter.convert()
  return tflite_model


FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('saved_model_path', '',
                                 'Path to input saved_model.')
tf.compat.v1.flags.DEFINE_string('tflite_model_path', '',
                                 'Path to output tflite module.')


def main(unused_argv):
  del unused_argv  # Unused.

  with open('model.tflite', 'wb') as fd:
    fd.write(saved_model_to_tflite(FLAGS.saved_model_path))


if __name__ == '__main__':
  FLAGS(sys.argv, known_only=True)
  FLAGS.unparse_flags()

  tf.compat.v1.app.run(main)
