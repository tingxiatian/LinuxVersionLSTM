#coding: utf-8
from __future__ import print_function

import yaml
import pickle
import codecs
import numpy as np
from collections import defaultdict
from utils import create_dictionary, load_embed_from_txt

import os

import pause

def build_vocabulary(path_data, path_vocs_dict, min_counts_dict, columns):
    # the columns set the feature count num of every dict

    print(path_data)

    print("building vocs...")
    file_data = codecs.open(path_data, "r", encoding="utf-8")
    line = file_data.readline()

    # count sequence length num
    sequence_length_dict = defaultdict(int)

    feature_item_dict_list = []
    for i in range(len(columns)):
        feature_item_dict_list.append(defaultdict(int))
    sequence_length = 0

    while line:
        line = line.strip()
        if not line:
            line = file_data.readline()
            sequence_length_dict[sequence_length] += 1
            sequence_length = 0
            continue
        items = line.split(" ")
        sequence_length += 1
        #print(items)
        for i in range(len(items)):
            feature_item_dict_list[i][items[i]] += 1
        line = file_data.readline()
    file_data.close()

    if sequence_length != 0:
        sequence_length_dict[sequence_length] += 1

    voc_sizes = []
    for i, name in enumerate(columns):
        size = create_dictionary(
            feature_item_dict_list[i], path_vocs_dict[name], start= 1,
            sort = True, min_count=min_counts_dict[name], overwrite=True
        )
        print("voc: %s, size: %d" % (path_vocs_dict[name], size))
        voc_sizes.append(size)
    print("length distribution of sentence :")
    #print(sorted(sequence_length_dict.items()))
    #print("done!")
    return voc_sizes, max(sequence_length_dict.keys())


def main():
    print("preprocessing...")
    with open("local_used_lstm_crf/new_config.yml", "r") as f:
        config = yaml.load(f)
    columns = config["model_params"]["feature_names"] + ["label"]
    min_counts_dict, path_vocs_dict = defaultdict(int), dict()
    feature_names = config["model_params"]["feature_names"]
    print("feature_names :")
    print(feature_names)
    for feature_name in feature_names:
        min_counts_dict[feature_name] = config["data_params"]["voc_params"][feature_name]["min_count"]
        path_vocs_dict[feature_name] = config["data_params"]["voc_params"][feature_name]["path"]
    path_vocs_dict["label"] = config["data_params"]["voc_params"]["label"]["path"]
    voc_sizes, sequence_length = build_vocabulary(
        path_data=config["data_params"]["path_train"],
        columns = columns,
        min_counts_dict=min_counts_dict,
        path_vocs_dict=path_vocs_dict
    )

    feature_dim_dict = dict()

    print("after init feature_dim_dict :")

    for i, feature_name in enumerate(feature_names):

        print("feature_name :{}".format(feature_name))

        # the embeding vec
        path_pre_train = config["model_params"]["embed_params"][feature_name]["path_pre_train"]
        if not path_pre_train:
            if i == 0:
                feature_dim_dict[feature_name] = 64
            else:
                feature_dim_dict[feature_name] = 32
            continue
        path_pkl = config["model_params"]["embed_params"][feature_name]["path"]
        path_vec = config["data_params"]["voc_params"][feature_name]["path"]

        with open(path_vec, "rb") as f:
            voc = pickle.load(f)
        print("load path_vec :{}".format(path_vec))


        #embedding_dict, vec_dim = load_embed_from_txt(path_pre_train)
        vec_dim = 64
        feature_dim_dict[feature_name] = vec_dim
        embedding_matrix = np.zeros((len(voc.keys()) + 1, vec_dim), dtype="float32")
        for item in voc:
            '''
            if item in embedding_dict:
                embedding_matrix[voc[item], :] = embedding_dict[item]
            else:
            '''
            embedding_matrix[voc[item], :] = np.random.uniform(-0.25, 0.25, size = (vec_dim))

        with open(path_pkl, "wb") as f:
            pickle.dump(embedding_matrix, f)
        print("the pickle path :")
        print(path_pkl)

    print("voc_sizes :" + "-" * 100)
    print(voc_sizes)

    label_size = voc_sizes[-1]
    voc_sizes = voc_sizes[:-1]

    config["model_params"]["nb_classes"] = label_size + 1

    for i, feature_name in enumerate(feature_names):
        # diff with ori
        config["model_params"]["embed_params"][feature_name]["shape"] = \
            [voc_sizes[i] + 1, feature_dim_dict[feature_name]]

    with codecs.open("new_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print("all done !")




if __name__ == "__main__":
    # test building voc

    main()