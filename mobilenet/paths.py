#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

CWD = os.path.abspath(__file__)


def get_mobilenet_path():
    return os.path.abspath(os.path.join(CWD, os.path.pardir))


def get_root_path():
    return os.path.abspath(os.path.join(get_mobilenet_path(), os.path.pardir))


def get_scripts_path():
    return os.path.abspath(os.path.join(get_root_path(), 'scripts'))


def get_imagenet_path():
    return os.path.abspath(os.path.join(get_root_path(), 'imagenet'))


def get_gloss_path():
    return os.path.abspath(os.path.join(get_imagenet_path(), 'gloss.txt'))


def get_label_map_proto_path():
    return os.path.abspath(os.path.join(get_imagenet_path(), 'imagenet_2012_challenge_label_map_proto.pbtxt'))


def get_synset_to_human_label_map_path():
    return os.path.abspath(os.path.join(get_imagenet_path(), 'imagenet_synset_to_human_label_map.txt'))


def get_words_mapping_path():
    return os.path.abspath(os.path.join(get_imagenet_path(), 'words.txt'))


def get_words_hierarchy_path():
    return os.path.abspath(os.path.join(get_imagenet_path(), 'wordnet.is_a.txt'))


def get_labels_path():
    return os.path.abspath(os.path.join(get_imagenet_path(), 'labels.txt'))


paths = {'root': get_root_path(),
         'mobilenet': get_mobilenet_path(),
         'imagenet': get_imagenet_path(),
         'scripts': get_scripts_path(),
         'gloss': get_gloss_path(),
         'label_map_proto': get_label_map_proto_path(),
         'synset_to_human_label_map': get_synset_to_human_label_map_path(),
         'words': get_words_mapping_path(),
         'hierarchy': get_words_hierarchy_path(),
         'labels': get_labels_path()
         }


if __name__ == '__main__':
    for k, p in paths.iteritems():
        print(k+": "+p)