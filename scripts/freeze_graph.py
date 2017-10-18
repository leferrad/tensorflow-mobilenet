from __future__ import print_function

from mobilenet.core import MobileNetDefaultFile, MobileNetV1Restored
from mobilenet.fileio import download_and_uncompress_tarball, get_logger
from mobilenet.imagenet import create_readable_names_for_imagenet_labels

import os
import re
import json
import argparse


def create_label_json_file(json_fn):
    labels = create_readable_names_for_imagenet_labels()

    with open(json_fn, 'w') as f:
        json.dump(labels, f,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))

    return labels

logger = get_logger(name="freeze_mobilenet", level='debug')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint-path', dest='checkpoint_path',
                        default='',
                        help="Path to directory with the checkpoints of the MobileNet model")
    args = parser.parse_args()

    #checkpoint_path = args.checkpoint_path
    checkpoint_path = '/home/leeandro04/Escritorio/NEW_mobilenet_v1_1.0_224_2017_06_14'

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

    checkpoint_file = model_pb.replace('.pb', '.ckpt')

    checkpoint_file = os.path.join(checkpoint_path, checkpoint_file)

    try:
        if os.path.exists(checkpoint_file+'.meta'):
            logger.info("Processing checkpoint file '%s' ...", checkpoint_file)
            mobilenet_model = MobileNetV1Restored(img_size=int(img_size), model_factor=float(factor))
            mobilenet_model.freeze_inference_graph(checkpoint_file, output_filename='frozen_graph.pb')
            create_label_json_file('/tmp/mobilenet/labels.json')
            #copyfile(os.path.join(model_dir, model_pb), os.path.join(img_subdir, model_pb))
        else:
            logger.info("Skipping not existing meta file '%s'...", checkpoint_file)
            pass
    except Exception as e:
        logger.error("Failed to process meta_file '%s': %s", checkpoint_file, str(e))