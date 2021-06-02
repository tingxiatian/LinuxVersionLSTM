#coding: utf-8
from __future__ import print_function
import codecs
from pprint import pprint
import pickle
import yaml


def generate_three_voc_id_dict():
    #train_text_path = "data/format_file.txt"
    train_text_path = "data/format_file_5.txt"
    with codecs.open(train_text_path, "r",
                     encoding="utf-8") as f:
        word_lines = f.readlines()

    nest_list = []
    for line in word_lines:
        if line.strip():
            split_col_list = line.strip().split(" ")
            nest_list.append(tuple(split_col_list))

    zip_list = zip(*nest_list)
    map_comnclusion = map(list ,map(set ,map(list ,zip_list)))

    f1_list, label_list = map_comnclusion

    def generator_id_dict_from_list(list_iter):
        return dict([(word, i) for i, word in enumerate(list_iter)])

    f1_dict, label_dict = map(generator_id_dict_from_list, [f1_list, label_list])

    with open("local_used_lstm_crf/new_config.yml") as f:
        config = yaml.load(f)

    print(config["data_params"]["voc_params"]["f1"]["path"])
    with open(config["data_params"]["voc_params"]["f1"]["path"], "wb") as f:
        pickle.dump(f1_dict, f)

    print(config["data_params"]["voc_params"]["label"]["path"])
    with open(config["data_params"]["voc_params"]["label"]["path"], "wb") as f:
        pickle.dump(label_dict, f)

    print("dump end")

def valid_ndarray():

    with open("data/embed/f1_embed.pkl", "rb") as f:
        f1_embed_array =pickle.load(f)

    with open("data/embed/f2_embed.pkl", "rb") as f:
        f2_embed_array =pickle.load(f)

    print(f1_embed_array.shape)
    print(f2_embed_array.shape)

    #print(f1_embed_array)
    #print(f2_embed_array)




if __name__ == "__main__":
    generate_three_voc_id_dict()