# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""TensorFlow Lite tooling helper functionality.

EXPERIMENTAL: APIs here are unstable and likely to change without notice.

@@toco_convert
@@toco_convert_protos

NOTE: this was adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/python/lite.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.contrib.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.contrib.lite.toco import types_pb2 as _types_pb2
from tensorflow.python.framework import dtypes as _dtypes

# Enum types from the protobuf promoted to the API
FLOAT = _types_pb2.FLOAT
INT32 = _types_pb2.INT32
INT64 = _types_pb2.INT64
STRING = _types_pb2.STRING
QUANTIZED_UINT8 = _types_pb2.QUANTIZED_UINT8
TENSORFLOW_GRAPHDEF = _toco_flags_pb2.TENSORFLOW_GRAPHDEF
TFLITE = _toco_flags_pb2.TFLITE
GRAPHVIZ_DOT = _toco_flags_pb2.GRAPHVIZ_DOT


def toco_convert_protos(model_flags_str, toco_flags_str, input_data_str, output_filename):
    """Convert `input_data_str` according to model and toco parameters.

    Unless you know what you are doing consider using
    the more friendly @{tf.contrib.lite.toco_convert}}.

    Args:
    model_flags_str: Serialized proto describing model properties, see
      `toco/model_flags.proto`.
    toco_flags_str: Serialized proto describing conversion properties, see
      `toco/toco_flags.proto`.
    input_data_str: Input data in serialized form (e.g. a graphdef is common)
    Returns:
    Converted model in serialized form (e.g. a TFLITE model is common).
    Raises:
    RuntimeError: When conversion fails, an exception is raised with the error
      message embedded.
    """
    # I need to do this import here, otherwise somehow it is not done outer this function
    from tensorflow.contrib.lite.toco.python import tensorflow_wrap_toco

    success = True

    try:
        output_str = tensorflow_wrap_toco.TocoConvert(model_flags_str, toco_flags_str, input_data_str)

        with open(output_filename, "wb") as f:
            f.write(output_str)
    except Exception as e:
        print("An error has occurred while converting model with TOCO!: %s" % str(e))
        success = False

    return success


def _tensor_name(x):
    return x.name.split(":")[0]


def toco_convert(input_data,
                 input_tensors,
                 output_tensors,
                 output_filename,
                 inference_type=FLOAT,
                 input_format=TENSORFLOW_GRAPHDEF,
                 output_format=TFLITE,
                 quantized_input_stats=None,
                 drop_control_dependency=True):
    """Convert a model using TOCO from `input_format` to `output_format`.

    Typically this is to convert from TensorFlow GraphDef to TFLite, in which
    case the default `input_format` and `output_format` are sufficient.

    Args:
    input_data: Input data (i.e. often `sess.graph_def`).
    input_tensors: List of input tensors. Type and shape are computed using
      `foo.get_shape()` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    inference_type: Currently must be `{FLOAT, QUANTIZED_UINT8}`.
    input_format: Type of data to read (currently must be TENSORFLOW_GRAPHDEF).
    output_format: Type of data to write (currently must be TFLITE or
      GRAPHVIZ_DOT)
    quantized_input_stats: For each member of input_tensors the mean and
      std deviation of training data. Only needed if `inference_type` is
      `QUANTIZED_UINT8`.
    drop_control_dependency: Drops control dependencies silently. This is due
      to tf lite not supporting control dependencies.

    Returns:
    The converted data. For example if tflite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

    Raises:
    ValueError: If the input tensor type is unknown
    RuntimeError: If TOCO fails to convert (in which case the runtime error's
      error text will contain the TOCO error log)
    """
    toco = _toco_flags_pb2.TocoFlags()
    toco.input_format = input_format
    toco.output_format = output_format
    toco.drop_control_dependency = drop_control_dependency
    model = _model_flags_pb2.ModelFlags()
    toco.inference_type = inference_type
    for idx, input_tensor in enumerate(input_tensors):
        if input_tensor.dtype == _dtypes.float32:
            tflite_input_type = FLOAT
        elif input_tensor.dtype == _dtypes.int32:
            tflite_input_type = INT32
        elif input_tensor.dtype == _dtypes.int64:
            tflite_input_type = INT64
        else:
            raise ValueError("Tensors %s not known type %r" % (input_tensor.name,
                                                               input_tensor.dtype))

        input_array = model.input_arrays.add()

        if inference_type == QUANTIZED_UINT8:
            if tflite_input_type == FLOAT:
                tflite_input_type = QUANTIZED_UINT8
            input_array.mean, input_array.std = quantized_input_stats[idx]

        input_array.name = _tensor_name(input_tensor)
        input_array.shape.dims.extend(map(int, input_tensor.get_shape()))
        toco.inference_input_type = tflite_input_type

    for output_tensor in output_tensors:
        model.output_arrays.append(_tensor_name(output_tensor))

    success = toco_convert_protos(model.SerializeToString(),
                                  toco.SerializeToString(),
                                  input_data.SerializeToString(),
                                  output_filename)
    return success
