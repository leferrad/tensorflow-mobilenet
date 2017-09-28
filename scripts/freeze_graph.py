from __future__ import print_function

from utils.fileio import download_and_uncompress_tarball, get_logger, is_valid_slim_directory, MobileNetDefaultFile

from tensorflow.python.framework import graph_util
import tensorflow as tf

import os
import re
from shutil import copyfile
from sys import path as syspath
import argparse

slim = tf.contrib.slim


def freeze_mobilenet(meta_file, img_size=224, model_factor=1.0, num_classes=1001):

    tf.reset_default_graph()

    inp = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3), name="input")
    weight_decay = 0.0
    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits, _ = mobilenet_v1.mobilenet_v1(inp, num_classes=num_classes,
                                              is_training=False,
                                              depth_multiplier=model_factor)

    predictions = tf.contrib.layers.softmax(logits)
    output = tf.identity(predictions, name='output')

    #tf.summary.FileWriter('/tmp/tf-logs/', graph=output.graph)

    ckpt_file = meta_file.replace('.meta', '')
    output_graph_fn = ckpt_file.replace('.ckpt', '.pb')
    output_node_names = "output"

    rest_var = slim.get_variables_to_restore()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        saver = tf.train.Saver(rest_var)
        saver.restore(sess, ckpt_file)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_fn, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        logger.info("%i ops in the final graph.", len(output_graph_def.node))


logger = get_logger(name="freeze_mobilenet", level='debug')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdir', '--slim-dir', dest='slim_dir', default='/root/tensorflow/models/research/slim',
                        help="Path to directory with TensorFlow's Slim models.")
    parser.add_argument('-ckpt', '--checkpoint-path', dest='checkpoint_path',
                        default='',
                        help="Path to directory with the checkpoints of the MobileNet model")
    args = parser.parse_args()

    slim_dir = args.slim_dir

    slim_dir_is_ok = slim_dir is not None and os.path.exists(slim_dir) and is_valid_slim_directory(slim_dir)

    if not slim_dir_is_ok:
        logger.error("Argument '-sdir' not valid! Please enter the correct path to Slim directory. "
                     "You can download it by: 'git clone https://github.com/tensorflow/models.git'")
        exit(0)

    # Adding this path to PYTHONPATH for importing some modules
    syspath.append(slim_dir)

    # Now we can import the modules from Slim
    # From tensorflow/models/slim
    from nets import mobilenet_v1

    checkpoint_path = args.checkpoint_path

    if checkpoint_path == '':
        # Then we have to download a default model
        checkpoint_path = '/tmp/mobilenet'
        logger.info("Checkpoint path is not valid. Setting default: %s", checkpoint_path)

        if not os.path.exists(checkpoint_path):
            logger.info("Creating non existing directory: %s", checkpoint_path)
            os.mkdir(checkpoint_path)

        # Model properties
        factor = MobileNetDefaultFile.MODEL_FACTOR
        img_size = MobileNetDefaultFile.IMG_SIZE

        logger.info("Setting default model properties...")


        model_dl = MobileNetDefaultFile.MODEL_DL_FMT.format(factor, img_size,
                                                            MobileNetDefaultFile.MODEL_DATE)

        logger.info("Setting default file to download: %s", model_dl)

        success = download_and_uncompress_tarball(checkpoint_path, filename=model_dl)

        if not success:
            logger.error("Model couldn't been downloaded and extracted. Exiting...")
            exit(0)

    else:
        if not os.path.exists(checkpoint_path):
            logger.error("Argument '-ckpt' not valid! Please enter the correct path to the MobileNet checkpoint. "
                         "You can omit this argument to allow the script to download a default model")
            exit(0)

        regex = r'mobilenet_v1_[0-9].[0-9]+_[0-9]+.*'

        match = False

        for fn in os.listdir(checkpoint_path):

            if re.match(regex, fn):
                numbers = re.findall(r'[0-9]+', fn)
                factor = numbers[1]+'.'+numbers[2]
                img_size = numbers[3]
                match = True
                break

        if not match:
            logger.error("Argument '-ckpt' doesn't have an expected format for the MobileNet model. "
                         "Please assert that the checkpoint file matches with the following expression: %s", str(regex))
            exit(0)

    logger.info("Model properties: factor=%s, img_size=%s", factor, img_size)

    model_pb = MobileNetDefaultFile.MODEL_PB_FMT.format(factor, img_size)

    meta_file = model_pb.replace('.pb', '.ckpt.meta')

    meta_file = os.path.join(checkpoint_path, meta_file)

    try:
        if os.path.exists(meta_file):
            logger.info("Processing meta_file '%s' ...", meta_file)
            freeze_mobilenet(meta_file, int(img_size), float(factor), num_classes=1001)
            #copyfile(os.path.join(model_dir, model_pb), os.path.join(img_subdir, model_pb))
        else:
            logger.info("Skipping not existing meta file '%s'...", meta_file)
            pass
    except Exception as e:
        logger.error("Failed to process meta_file '%s': %s", meta_file, str(e))