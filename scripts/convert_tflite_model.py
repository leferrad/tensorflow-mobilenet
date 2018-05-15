from __future__ import print_function

"""Script to convert a model in Protobuf format into a one in TFLite format"""

from mobilenet.core import MobileNetV1Restored
from mobilenet.fileio import get_logger

import os
import argparse


logger = get_logger(name="make_inferences", level='debug')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pb', '--pb-path', dest='pb_path', #required=True,
                        help="Path to inference graph in Protobuf format (i.e. file ending with .pb)")
    parser.add_argument('-out', '--out-fn', dest='output_filename', #required=True,
                        help="Path to output filename for the resulting model converted")
    args = parser.parse_args()

    pb_path = args.pb_path
    output_filename = args.output_filename

    if not os.path.exists(pb_path):
        logger.error("Argument '-pb' not valid! Please enter the correct path to the MobileNet protobuf model.")
        exit(0)

    try:
        logger.info("Processing pb_file '%s' ...", pb_path)
        mobilenet_model = MobileNetV1Restored(img_size=224, model_factor=1.0)
        _ = mobilenet_model.restore_session_from_frozen_graph(filename=pb_path)

        success = mobilenet_model.convert_tflite_format(output_filename=output_filename)

        if success:
            logger.info("Conversion achieved successfully! Model was saved in '%s'", output_filename)
        else:
            logger.error("Conversion to TF Lite format couldn't been achieved. "
                         "Please check the logs for more information")

    except Exception as e:
        logger.error("An error has ocurred! %s", str(e))
