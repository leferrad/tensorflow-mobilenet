from __future__ import print_function

from utils.fileio import get_logger, load_json_as_dict

from tensorflow.python.platform import gfile
import tensorflow as tf

from scipy.misc import imread, imresize
import numpy as np

import os
import time
import argparse

slim = tf.contrib.slim


logger = get_logger(name="make_inferences", level='debug')


def load_labels():
    path = '../labels.json'
    labels = load_json_as_dict(path)
    return labels


def read_tensor_from_image_file(file_name, input_height=224, input_width=224, input_mean=-127, input_std=127):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def report_output(img_filename, prediction, labels, n_top=10):
    logger.info("Top %i predictions for the image given by '%s':", n_top, img_filename)
    top_predictions = sorted(enumerate(prediction), key=lambda (i, p): -p)[:n_top]
    c = 1
    for i, p in top_predictions:
        l = labels[str(i)]
        logger.info("%i. %s (prob=%.5f)", c, l, p)
        c += 1


def load_and_prepare_image(filename, img_size=224):
    """
    NOTE: don't use this!!

    :param filename:
    :param img_size:
    :return:
    """
    #img = np.asarray(Image.open(filename).convert('RGB'))
    img = imread(filename, flatten=False)
    img = imresize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)

    # Preprocess
    img = (img - 127.) / 127.

    return img

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="mobilenet",
            op_dict=None,
            producer_op_list=None
        )

    return graph


def load_mobilenet(model_filename, images_path, logs_directory='/tmp/tf-logs'):

    # Now load model and make inferences
    predictions = []
    labels = load_labels()

    graph = load_graph(model_filename)

    tf.summary.FileWriter(logs_directory, graph=graph)

    with tf.Session(graph=graph) as sess:

        input_tensor = graph.get_tensor_by_name("mobilenet/input:0")
        softmax_tensor = graph.get_tensor_by_name("mobilenet/MobilenetV1/Predictions/Reshape_1:0")
        avg_pool_tensor = graph.get_tensor_by_name("mobilenet/MobilenetV1/Logits/AvgPool_1a/AvgPool:0")

        for im in os.listdir(images_path):
            tic = time.time()
            img = load_and_prepare_image(os.path.join(images_path, im))
            #img = read_tensor_from_image_file(os.path.join(images_path, im))

            prediction, embedding = sess.run([softmax_tensor, avg_pool_tensor], {input_tensor: img})
            prediction = np.squeeze(prediction)
            embedding = np.squeeze(embedding)


            report_output(im, prediction, labels=labels, n_top=10)

            toc = time.time()

            print("Detection made in %.2f sec" % (toc-tic))

            predictions.append((im, embedding))

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pb', '--pb-path', dest='pb_path', #required=True,
                        help="Path to inference graph in Protobuf format (i.e. file ending with .pb)")
    parser.add_argument('-img', '--img', dest='img_path', #required=True,
                        help="Path to directory with images to use for predictions")
    args = parser.parse_args()

    #pb_path = args.pb_path

    pb_path = '/home/leeandro04/Escritorio/NEW_mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_final.pb'
    img_path = '/home/leeandro04/work/poc_lbp_pic/img/whatsapp/'
    #img_path = '/home/leeandro04/Escritorio/imgs/'

    if not os.path.exists(pb_path):
        logger.error("Argument '-pb' not valid! Please enter the correct path to the MobileNet protobuf model.")
        exit(0)

    try:
        logger.info("Processing meta_file '%s' ...", pb_path)
        embeddings = load_mobilenet(pb_path, img_path)

        print("DONE")

    except Exception as e:
        logger.error("Failed to process pb_file '%s': %s", pb_path, str(e))

