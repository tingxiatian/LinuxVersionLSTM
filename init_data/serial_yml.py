#coding: utf-8
import pickle
import yaml

def serial_func():
    with open("/Users/svjack/IdeaProjects/tensorflowLearn/LinuxVersionLSTM-CRF/local_used_lstm_crf/new_config.yml", "r") as f:
        config_dict = yaml.load(f)
    with open("config_dict.pkl", "wb") as f:
        pickle.dump(config_dict, f)

if __name__ == "__main__":
    serial_func()