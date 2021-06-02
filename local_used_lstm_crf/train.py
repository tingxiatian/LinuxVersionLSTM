#coding: utf-8
from __future__ import print_function

# the yaml load to dict construct
import yaml
import pickle
from load_data import load_vecs, init_data
from model import SequenceLabelModel

def main():

    yaml_path = "local_used_lstm_crf/new_config.yml"
    with open(yaml_path, "r") as f:
        config = yaml.load(f)

    # return in the first loop
    #return
        
    feature_names = config["model_params"]["feature_names"]
    
    feature_weight_shape_dict, feature_weight_dropout_dict, feature_init_weight_dict = dict(), dict(), dict()
    for feature_name in feature_names:
        feature_weight_shape_dict[feature_name] = config["model_params"]["embed_params"][feature_name]["shape"]
        feature_weight_dropout_dict[feature_name] = config["model_params"]["embed_params"][feature_name]["dropout_rate"]
        
        path_pre_train = config["model_params"]["embed_params"][feature_name]["path"]
        #path_pre_train = ""
        if path_pre_train:
            with open(path_pre_train, "rb") as f:
                feature_init_weight_dict[feature_name] = pickle.load(f)
    
    # load vecs
    path_vocs = []
    for feature_name in feature_names:
        path_vocs.append(config["data_params"]["voc_params"][feature_name]["path"])
    path_vocs.append(config["data_params"]["voc_params"]["label"]["path"])
    vocs = load_vecs(path_vocs)
    
    # load train data
    sep_str = config["data_params"]["sep"]
    assert sep_str in ["table", "space"]
    sep = "\t" if sep_str == "table" else ' '
    data_dict = init_data(
        path = config["data_params"]["path_train"],
        feature_names = feature_names,
        sep = sep,
        vocs = vocs,
        max_len= config["model_params"]["sequence_length"],
        model = "train"
    )
    
    
    model = SequenceLabelModel(
        sequence_length=config["model_params"]["sequence_length"],
        nb_classes=config["model_params"]["nb_classes"],
        nb_hidden = config["model_params"]["bilstm_params"]["num_units"],
        
        feature_weight_shape_dict = feature_weight_shape_dict,
        feature_init_weight_dict = feature_init_weight_dict,
        feature_weight_dropout_dict = feature_weight_dropout_dict,
        
        dropout_rate = config["model_params"]["dropout_rate"],
        nb_epoch= config["model_params"]["nb_epoch"],
        feature_names = feature_names,
        
        batch_size=config["model_params"]["batch_size"],
        train_max_patience = config["model_params"]["max_patience"],
        use_crf = config["model_params"]["use_crf"],
        l2_rate=config["model_params"]["l2_rate"],
        rnn_unit = config["model_params"]["rnn_unit"],
        learning_rate= config["model_params"]["learning_rate"],
        clip = config["model_params"]["clip"],
        path_model= config["model_params"]["path_model"]
    )
    
    model.fit(
        data_dict= data_dict, dev_size=config["model_params"]["dev_size"]
    )


if __name__ == "__main__":
    main()