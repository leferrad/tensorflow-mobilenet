from __future__ import print_function

from mobilenet.core import MobileNetV1Restored
from mobilenet.fileio import get_logger
from mobilenet.imagenet import LabelLookup

import os
import argparse


logger = get_logger(name="make_inferences", level='debug')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pb', '--pb-path', dest='pb_path', required=True,
                        help="Path to inference graph in Protobuf format (i.e. file ending with .pb)")
    parser.add_argument('-img', '--img', dest='img_path', required=True,
                        help="Path to directory with images to use for predictions")
    args = parser.parse_args()

    pb_path = args.pb_path
    img_path = args.img_path

    if not os.path.exists(pb_path):
        logger.error("Argument '-pb' not valid! Please enter the correct path to the MobileNet protobuf model.")
        exit(0)

    try:
        logger.info("Processing pb_file '%s' ...", pb_path)
        mobilenet_model = MobileNetV1Restored(img_size=224, model_factor=1.0)
        _ = mobilenet_model.restore_session_from_frozen_graph(filename=pb_path)
        predictions = mobilenet_model.predict_on_images(img_path)
        label_lookup = LabelLookup()

        for fn, prediction in predictions.items():

            top_predictions = sorted(enumerate(prediction), key=lambda (i, p): -p)[:10]
            top_predictions = [(label_lookup.id_to_string(i), p) for (i, p) in top_predictions]

            logger.info("Top %i predictions for the image given by '%s':", 10, fn)
            c = 1
            for l, p in top_predictions:
                logger.info("%i. %s (prob=%.5f)", c, l, p)
                c += 1
            logger.info("\n")

        logger.info("DONE")

    except Exception as e:
        logger.error("Failed to process pb_file '%s': %s", pb_path, str(e))

