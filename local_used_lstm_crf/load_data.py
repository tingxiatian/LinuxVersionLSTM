#coding: utf-8
from __future__ import print_function

from pprint import pprint

import codecs
import pickle
import numpy as np
from utils import map_item2id


def load_vecs(paths):
    vocs = []
    for path in paths:
        with open(path, "rb") as f:
            vocs.append(pickle.load(f))

    return vocs

def load_lookup_tables(paths):
    lookup_tabels = []
    for path in paths:
        with open(path, "rb", encoding = "utf-8") as f:
            lookup_tabels.append(pickle.load(f))

    return lookup_tabels

def init_data(path, feature_names, vocs, max_len, model = "train", sep= " "):
    assert model in ("train", "test")
    f = codecs.open(path, "r", encoding= "utf-8")
    sentences = f.read().strip().split("\n\n")
    sentences_count = len(sentences)
    feature_count = len(feature_names)
    data_dict = dict()
    for feature_name in feature_names:
        data_dict[feature_name] = np.zeros(
            shape=(sentences_count, max_len), dtype="int32"
        )
    if model == "train":
        data_dict["label"] = np.zeros(
            shape= (sentences_count, max_len), dtype = "int32"
        )

    for index, sentence in enumerate(sentences):
        items = sentence.split("\n")
        one_instance_items = []

        # one_instance_items is longer than the total length of feature name.
        # the last feature_token is the mark of the word.
        # store in the last list
        [one_instance_items.append([]) for _ in range(len(feature_names) + 1)]
        for item in items:
            feature_tokens = item.split(" ")

            for j in range(feature_count):
                one_instance_items[j].append(feature_tokens[j])
            if model == "train":
                one_instance_items[-1].append(feature_tokens[-1])

        # use map_item2id to map true word to index from vocs
        for i in range(len(feature_names)):
            data_dict[feature_names[i]][index, :] = map_item2id(
                one_instance_items[i], vocs[i], max_len
            )

        if model == "train":
            data_dict["label"][index, :] = map_item2id(
                one_instance_items[-1], vocs[-1], max_len
            )
    f.close()

    return data_dict


if __name__ == "__main__":
    # test serials
    feature_names = ["f1", "f2"]
    one_instance_items = []
    [one_instance_items.append([]) for _ in range(len(feature_names) + 1)]
    print("one_instance_items :")
    pprint(one_instance_items)

