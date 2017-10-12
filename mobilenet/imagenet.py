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

from mobilenet.fileio import load_json_as_dict, save_dict_as_json, get_logger
from mobilenet.paths import paths

from six.moves import urllib

import os
import re

NUM_CLASSES = 1001
DATA_BASE_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
SYNSET_URL = '{}/imagenet_lsvrc_2015_synsets.txt'.format(DATA_BASE_URL)
SYNSET_TO_HUMAN_URL = '{}/imagenet_metadata.txt'.format(DATA_BASE_URL)


logger = get_logger(name=__name__, level='debug')


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


class LabelManager(object):
    def __init__(self):
        paths_to_load = ['labels', 'words', 'hierarchy', 'synset_to_human_label_map']
        self.paths = dict(filter(lambda (k, p): p in paths_to_load, paths.items()))
        self.label_lookup = None
        self.load()

    @staticmethod
    def _parse_2col_txt_into_dict(filename):
        lines_txt = open(filename, 'r').readlines()
        result = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in lines_txt:
            parsed_items = p.findall(line)
            k = parsed_items[0]
            v = parsed_items[2]
            result[k] = v

        return result

    def load(self):
        # Asserting existence of all used paths
        for p in self.paths.values():
            if not os.path.exists(p):
                logger.error('File does not exist %s', p)
                return

        # Loads mapping from string UID to human-readable string
        self.uid_to_human = self._parse_2col_txt_into_dict(self.paths['synset_to_human_label_map'])

        # Loads the final mapping of integer node ID to human-readable string
        self.node_id_to_name = {}
        for key, val in self.node_id_to_uid.items():
            if val not in self.uid_to_human:
                logger.error('Failed to locate: %s', val)
            name = self.uid_to_human[val]
            self.node_id_to_name[key] = name

        # Finally, add the class 0
        self.node_id_to_name[0] = 'background'

    def id_to_string(self, node_id):
        return self.node_id_to_name.get(node_id, '')

    def __getitem__(self, item):
        return self.id_to_string(item)


class LabelLookup(object):
    def __init__(self, path_to_labels=None):
        if path_to_labels is None:
            path_to_labels = paths['labels']

        self.path = path_to_labels

        if not os.path.exists(self.path):
            logger.error("Path to labels '%s' does not exist!", self.path)
            return

        self.label_lookup = {}

        # Loads the mapping of integer node ID to human-readable string
        with open(self.path, 'r') as lines_txt:
            for i, line in enumerate(lines_txt):
                self.label_lookup[i] = line.strip()

    def id_to_string(self, i):
        return self.label_lookup.get(i, None)

    def __getitem__(self, item):
        return self.id_to_string(item)


class WNIDLookup(object):
    def __init__(self, path_to_words=None, cache=False):
        if path_to_words is None:
            path_to_words = paths['words']

        self.path = path_to_words

        if not os.path.exists(self.path):
            logger.error("Path to words mapping '%s' does not exist!", self.path)
            return

        self.flag_cache = cache
        self.wnid_lookup = {}
        self.regex = re.compile(r'[n\d]*[ \S,]*')

        if self.flag_cache:
            self.load_words()

    def generator_lines(self):
        with open(self.path, 'r') as lines:
            for line in lines:
                parsed_items = self.regex.findall(line)
                if parsed_items:
                    yield parsed_items[0], parsed_items[2]

    def load_words(self):
        for k, v in self.generator_lines():
            self.wnid_lookup[k] = v

    def get_label_from_wnid(self, wnid):
        if self.flag_cache:
            return self.wnid_lookup.get(wnid, None)
        else:
            for k, v in self.generator_lines():
                if k == wnid:
                    return v

    def get_wnid_from_label(self, label):
        if self.flag_cache:
            result = None
            for wnid, lab in self.wnid_lookup.items():
                if lab == label:
                    result = wnid
                    break
            return result

        else:
            for k, v in self.generator_lines():
                if v == label:
                    return k


class HierarchyLookup(object):
    def __init__(self, path_to_hierarchy=None, cache=True):
        if path_to_hierarchy is None:
            path_to_hierarchy = paths['hierarchy']

        self.path = path_to_hierarchy

        if not os.path.exists(self.path):
            logger.error("Path to hierarchy file '%s' does not exist!", self.path)
            return

        self.flag_cache = cache
        self.parent_lookup = {}
        self.regex = re.compile(r'[n\d]*[ \S,]*')

        if self.flag_cache:
            self.load_hierarchy_lookup()

    def generator_lines(self):
        with open(self.path, 'r') as lines:
            for line in lines:
                yield line.strip().split(' ')

    def load_hierarchy_lookup(self):
        for w1, w2 in self.generator_lines():
            if w2 not in self.parent_lookup:
                self.parent_lookup[w2] = [w1]
            else:
                self.parent_lookup[w2].append(w1)

    def get_full_hierarchy(self, item, flat=True, depth=-1):
        result = [tuple([item])]
        item = tuple([item])
        d = 0
        if depth == -1:
            depth = float("inf")
        while d < depth:
            parents = reduce(lambda a, b: a+b, [self.parent_lookup.get(it, []) for it in item])
            if len(parents) == 0:
                break
            else:
                result = [tuple(parents)]+result
                item = parents
                d += 1

        if flat:
            result = reduce(lambda a, b: a+b, map(lambda x: list(x), result))

        return result



if __name__ == '__main__':
    node_lookup = LabelLookup()
    print("Labels: %s" % str(node_lookup.label_lookup.items()))

    wnid = 'n02124075'

    hier_lookup = HierarchyLookup()
    hier = hier_lookup.get_full_hierarchy(wnid, flat=True, depth=5)


    wnid_lookup = WNIDLookup(cache=True)

    for w in reversed(hier):
        print("wnid=%s, label=%s" % (w, wnid_lookup.get_label_from_wnid(w)))