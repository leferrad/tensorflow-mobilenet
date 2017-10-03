# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapted from https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet.py"""

from mobilenet.fileio import load_json_as_dict, save_dict_as_json

from six.moves import urllib

import os

NUM_CLASSES = 1001
DATA_BASE_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
SYNSET_URL = '{}/imagenet_lsvrc_2015_synsets.txt'.format(DATA_BASE_URL)
SYNSET_TO_HUMAN_URL = '{}/imagenet_metadata.txt'.format(DATA_BASE_URL)


def create_readable_names_for_imagenet_labels():
    """Create a dict mapping label id to human readable string.
    Returns:
      labels_to_names: dictionary where keys are integers from to 1000
      and values are human-readable names.
    We retrieve a synset file, which contains a list of valid synset labels used
    by ILSVRC competition. There is one synset one per line, eg.
          #   n01440764
          #   n01443537
    We also retrieve a synset_to_human_file, which contains a mapping from synsets
    to human-readable names for every synset in Imagenet. These are stored in a
    tsv format, as follows:
          #   n02119247    black fox
          #   n02119359    silver fox
    We assign each synset (in alphabetical order) an integer, starting from 1
    (since 0 is reserved for the background class).
    Code is based on
    https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py#L463
    """

    # pylint: disable=g-line-too-long

    filename, _ = urllib.request.urlretrieve(SYNSET_URL)
    synset_list = [s.strip() for s in open(filename).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    filename, _ = urllib.request.urlretrieve(SYNSET_TO_HUMAN_URL)
    synset_to_human_list = open(filename).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    # There are 1000 classes plus the 0 class (background)
    labels_to_names = {0: 'background'}

    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


def load_imagenet_labels():
    path_to_labels = '../labels.json'

    labels = None

    if os.path.exists(path_to_labels):
        # return: dict
        labels = load_json_as_dict(path_to_labels)

        # dict to list to allow compatibility with the other func
        labels = [labels[k] for k in sorted(labels.keys(), key=lambda k: int(k))]

    if labels is None:
        # return: list
        labels = create_readable_names_for_imagenet_labels()
        save_dict_as_json(labels, filename=path_to_labels, pretty_print=True)

    return labels