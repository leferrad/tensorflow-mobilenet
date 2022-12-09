#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import logging.handlers
import tarfile
import os
import json
from sys import stdout
from six.moves import urllib


def get_logger(name='mobilenet', level='debug'):
    """
    Function to obtain a normal logger

    :param name: string
    :param level: string, which can be 'info' or 'debug'
    :return: logging.Logger
    """

    levels = {'info': logging.INFO,
              'debug': logging.DEBUG}

    # If the level is not supported, then force it to be info
    if level not in levels:
        level = 'info'
    level = levels[level]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger

logger = get_logger(name=__name__, level='debug')


def compress_tar_files(files, filename):
    if isinstance(files, list) is False:
        files = [files]

    try:
        with tarfile.open(filename, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=os.path.basename(f))
        successful = True
    except:
        successful = False
    return successful


def uncompress_tar_files(filename):
    try:
        with tarfile.open(filename, "r:*") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=os.path.dirname(filename))
        successful = True
    except:
        successful = False
    return successful


class MobileNetDefaultFile(object):
    MODEL_REPO_BASE_URL = 'http://download.tensorflow.org/models/'
    MODEL_DATE = '2017_06_14'
    MODEL_BASE_FMT = 'mobilenet_v1_{}_{}'
    MODEL_DL_FMT = MODEL_BASE_FMT + '_{}.tar.gz'
    MODEL_PB_FMT = MODEL_BASE_FMT + '.pb'
    MODEL_FACTOR = '0.50'
    IMG_SIZE = 224


def download_and_uncompress_tarball(data_dir, filename):

    def _progress(count, block_size, total_size):
        stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        stdout.flush()

    tarball_url = os.path.join(MobileNetDefaultFile.MODEL_REPO_BASE_URL, filename)
    filepath = os.path.join(data_dir, filename)

    success = False

    if not os.path.exists(filepath):
        try:
            logger.info("Downloading file from %s ...", tarball_url)
            filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            logger.info("Successfully downloaded '%s': %s bytes.", filename, str(statinfo.st_size))
            success = True
        except Exception as e:
            logger.info("Error occurred while downloading model: %s", str(e))
    else:
        logger.info("Tarball '%s' already exists -- not downloading", filename)
        success = True

    if success:
        logger.info("Extracting tarball '%s'...", filename)
        success = uncompress_tar_files(filepath)

    return success


def is_valid_slim_directory(slim_directory):
    """
    An easy way to figure out if the TensorFlow Slim's directory is correct or not,
      by comparing their sub-folders with the expected ones.

    :param slim_directory: str
    :return: boolean
    """
    expected_subfolders = ['nets', 'preprocessing', 'datasets']  # The most important ones
    actual_subfolders = os.listdir(slim_directory)
    return all([f in actual_subfolders for f in expected_subfolders])


def save_dict_as_json(dictobj, filename, pretty_print=True):
    try:
        with open(filename, 'w') as f:
            if pretty_print is True:
                json.dump(dictobj, f, sort_keys=True, indent=4)
            else:
                json.dump(dictobj, f)
        successful = True
    except:
        successful = False
    return successful


def load_json_as_dict(filename):
    try:
        with open(filename, 'r') as f:
            dictobj = json.load(f)
    except:
        dictobj = None
    return dictobj