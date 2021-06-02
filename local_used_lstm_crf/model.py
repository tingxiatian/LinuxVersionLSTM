#coding: utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math
from utils import uniform_tensor, get_sequence_actual_length, zero_nil_slot, shuffle_matrix
from tqdm import tqdm

#from pprint import pprint

class SequenceLabelModel(object):

    # the feature weight and shape dict setting the input conatructure in
    # bi-lstm
    # the train_max_patience setting the un promote loss step.
    # clip the gradient to avoid grad explosion

    # path_model: the model path for serial model
    # the setting of multi features will be explained following

    # the feature_weight dict set in the utils to perform word embeddings(may be word2vec)

    def __init__(self, sequence_length, nb_classes, nb_hidden = 512, feature_names = None,
                 feature_init_weight_dict = None, feature_weight_shape_dict = None,
                 feature_weight_dropout_dict = None, dropout_rate = 0., use_crf = True,
                 path_model = None, nb_epoch = 200, batch_size = 128, train_max_patience = 10,
                 l2_rate = 0.01, rnn_unit = "lstm", learning_rate = 0.001, clip = None
                 ):
        self.sequence_length = sequence_length
        self.nb_classes = nb_classes
        self.nb_hidden = nb_hidden

        self.feature_names = feature_names
        self.feature_init_weight_dict = feature_init_weight_dict if \
            feature_init_weight_dict else dict()
        self.feature_weight_shape_dict = feature_weight_shape_dict
        self.feature_weight_dropout_dict = feature_weight_dropout_dict

        self.dropout_rate = dropout_rate
        self.use_crf = use_crf
        self.path_model = path_model

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.train_max_patience = train_max_patience

        self.l2_rate = l2_rate
        self.rnn_unit = rnn_unit
        self.learning_rate = learning_rate
        self.clip = clip

        # temporarily use none for avoid error
        self.clip = None

        assert len(feature_names) == len(list(set(feature_names))), \
            "duplication of feature names"

        self.build_model()


    def build_model(self):
        self.input_feature_ph_dict = dict()
        self.weight_dropout_ph_dict = dict()
        self.feature_weight_dict = dict()
        self.nil_vars = set()
        self.dropout_rate_ph = tf.placeholder(tf.float32, name = "dropout_rate_ph")

        # label ph
        # the input label ph shape [batch-size, sequence_length]
        # this is same as labeled word as y (output)
        self.input_label_ph = tf.placeholder(
            dtype=tf.int32, shape = [None, self.sequence_length], name = "input_label_ph"
        )

        for feature_name in self.feature_names:

            # the feature name have the same shaoe as input_label_ph
            # this is same as x (input_feature)
            self.input_feature_ph_dict[feature_name] = tf.placeholder(
                dtype= tf.int32, shape=[None, self.sequence_length],
                name = "input_feature_ph_%s" % feature_name
            )

            # dropout rate ph not setting shape can feed with any shape tensor
            self.weight_dropout_ph_dict[feature_name] = tf.placeholder(
                tf.float32, name = "dropout_ph_%s" % feature_name
            )

            # init feature weights
            if feature_name not in self.feature_init_weight_dict:
                feature_weight = uniform_tensor(
                    shape=self.feature_weight_shape_dict[feature_name],
                    name = "f_w_%s" % feature_name
                )
                self.feature_weight_dict[feature_name] = tf.Variable(
                    initial_value=feature_weight, name = "feature_weight_%s" % feature_name
                )
            else:
                self.feature_weight_dict[feature_name] = tf.Variable(
                    initial_value=self.feature_init_weight_dict[feature_name],
                    name = "feature_weight_%s" % feature_name
                )

            # add feature weight name to nil_vars. record the given feature weights
            # the correspond first row gradient will be zero.
            self.nil_vars.add(self.feature_weight_dict[feature_name].name)

            # init not dropout prob to 0
            if feature_name not in self.feature_weight_dropout_dict:
                self.feature_weight_dropout_dict[feature_name] = .0

        # embedding
        # this use the embedding drop
        self.embedding_features = []
        for feature_name in self.feature_names:
            embedding_features = tf.nn.dropout(
                tf.nn.embedding_lookup(
                    self.feature_weight_dict[feature_name],
                    ids=self.input_feature_ph_dict[feature_name],
                    name = "embedding_feature_%s" % feature_name
                ),
                keep_prob= 1.0 - self.weight_dropout_ph_dict[feature_name],
                name = "embedding_feature_dropout_%s" % feature_name
            )
            self.embedding_features.append(embedding_features)

        #concat all features
        #after embedding lookup the total dim is up to 3
        input_features = self.embedding_features[0] if len(self.embedding_features) == 1\
            else tf.concat(values=self.embedding_features, axis = 2, name = "input_features")

        # bi-lstm
        if self.rnn_unit == "lstm":
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nb_hidden, forget_bias=1.0, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nb_hidden, forget_bias=1.0, state_is_tuple=True)
        elif self.rnn_unit == "gru":
            fw_cell = tf.nn.rnn_cell.GRUCell(self.nb_hidden)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.nb_hidden)
        else:
            raise ValueError("rnn_unit must in (lstm gru)!")

        # use feature_names[0]
        # it will feed self.input_feature_ph_dict with identity batch data
        # so seek in non-zero indexes to first feature_names works.
        self.sequence_actual_length = get_sequence_actual_length(
            self.input_feature_ph_dict[self.feature_names[0]]
        )

        # para rough same as dynamic_rnn use sequence_length(non-padding list length)
        # too setting the input format. contrast to static version with padding.
        # same as rnn return list output states as second tuple element.
        # rnn_outputs is a tuple (output_fw, output_bw)
        '''
        output_fw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_fw.output_size]`
        and output_bw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_bw.output_size]`.
        '''
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, input_features, scope="bi-lstm",
            dtype=tf.float32, sequence_length=self.sequence_actual_length
        )

        # concat in axis cell.output_size
        # output shape [batch_size, max_time, sum_output_size]
        lstm_output = tf.nn.dropout(
            tf.concat(rnn_outputs, axis=2, name="lstm_output"),
            keep_prob= 1.0 - self.dropout_rate_ph, name = "lstm_output_dropout"
        )

        # softmax
        self.outputs = tf.reshape(lstm_output, [-1, self.nb_hidden * 2], name = "outputs")
        self.softmax_w = tf.get_variable("softmax_w", [self.nb_hidden * 2, self.nb_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.nb_classes])

        self.logits = tf.reshape(
            tf.matmul(self.outputs, self.softmax_w) + self.softmax_b,
            shape = [-1, self.sequence_length, self.nb_classes],
            name = "logits"
        )

        # compute los
        self.loss = self.compute_loss()
        self.l2_loss = self.l2_rate * (tf.nn.l2_loss(self.softmax_w) + tf.nn.l2_loss(self.softmax_b))
        self.total_loss = self.loss + self.l2_loss

        # train op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # retrieve grads and vars to perform grads clip rather than deliver the
        # grads directly

        # compute grad may with error:

        grads_and_vars = optimizer.compute_gradients(self.total_loss)

        #print("after grads_and_vars init")

        nil_grads_and_vars = []

        # this loop choose the name in self.nil_vars
        # fill the first row of weight grad to be zero
        # but the reason
        # ??????????
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        global_step = tf.Variable(0, name = "global_step", trainable=False)
        if self.clip:
            print("error occur in build process :")
            print("nil_grads_and_vars :")
            #print(nil_grads_and_vars)
            for ele in nil_grads_and_vars:
                print(ele)

            gradients, variables = zip(*nil_grads_and_vars)
            gradients, _ = tf.clip_by_average_norm(gradients, self.clip)

            grad_summary = tf.summary.histogram("grad", gradients)

            self.train_op = optimizer.apply_gradients(
                zip(gradients, variables), name = "train_op", global_step=global_step
            )
        else:
            self.train_op = optimizer.apply_gradients(
                nil_grads_and_vars, name = "train_op", global_step=global_step
            )

        # the setting of gpu list can change (maybe the device index)
        gpu_options = tf.GPUOptions(visible_device_list = '0', allow_growth = True)
        self.sess = tf.Session(config= tf.ConfigProto(gpu_options = gpu_options))

        # var init
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self, data_dict, dev_size = 0.2, seed = 1337):
        data_train_dict, data_dev_dict = self.split_train_dev(data_dict, dev_size = dev_size)
        self.saver = tf.train.Saver()
        train_data_count = data_train_dict["label"].shape[0]

        # the iter step in every epoch
        nb_train = int(math.ceil(train_data_count / float(self.batch_size)))
        # the min loss setting if the loss smaller than and stop
        min_dev_loss = 1000
        current_patience = 0

        for step in range(self.nb_epoch):
            print("Epoch %d / %d" % (step + 1, self.nb_epoch))

            data_list = [data_train_dict["label"]]

            [data_list.append(data_train_dict[name]) for name in self.feature_names]
            shuffle_matrix(*data_list, seed=seed)

            # train
            train_loss = 0.
            #for i in tqdm(range(nb_train)):
            for i in range(nb_train):
                feed_dict = dict()
                batch_indices = np.arange(i * self.batch_size, (i + 1) * self.batch_size)\
                if (i + 1) * self.batch_size <= train_data_count else np.arange(i * self.batch_size, train_data_count)

                for feature_name in self.feature_names:
                    # feature
                    batch_data = data_train_dict[feature_name][batch_indices]
                    item = {self.input_feature_ph_dict[feature_name]: batch_data}
                    feed_dict.update(item)

                    # dropout
                    dropout_rate = self.feature_weight_dropout_dict[feature_name]
                    item = {self.weight_dropout_ph_dict[feature_name]: dropout_rate}
                    feed_dict.update(item)

                # the tail drop_out feed
                feed_dict.update({self.dropout_rate_ph: self.dropout_rate})

                # label feed
                batch_label = data_train_dict["label"][batch_indices]
                feed_dict.update({self.input_label_ph: batch_label})

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)
                train_loss += loss
            train_loss /= float(nb_train)

            # compute loss on dev datasets on every epoch
            dev_loss = self.evaluate(data_dev_dict)

            print("train loss: %f, dev loss: %f" % (train_loss, dev_loss))

            if not self.path_model:
                continue
            if dev_loss < min_dev_loss:
                current_patience = 0

                self.saver.save(self.sess, self.path_model)
                print("model has saved to %s" % self.path_model)

            else:
                current_patience += 1
                print("no improvement, current patience %d / %d" % (current_patience, self.train_max_patience))
                if self.train_max_patience and current_patience >= self.train_max_patience:
                    print("\nfinish training! (early stopping, max patience: %d)" % self.train_max_patience)
                    return

        print("\nfinish training!")
        return


    def predict(self, data_test_dict):
        print("predicting...")

        data_count = data_test_dict["label"].shape[0]
        nb_test = int(math.ceil(data_count / float(self.batch_size)))
        viterbi_sequences = []
        for i in tqdm(range(nb_test)):
            feed_dict = dict()
            batch_indices = np.arange(i * self.batch_size, (i + 1) * self.batch_size)\
            if (i + 1) * self.batch_size <= data_count else np.arange(i * self.batch_size, data_count)

            for feature_name in self.feature_names:
                item = {self.input_feature_ph_dict[feature_name] :data_test_dict[feature_name][batch_indices]}
                feed_dict.update(item)

                # dropout
                feed_dict.update({self.weight_dropout_ph_dict[feature_name] : 0.0})

            feed_dict.update({self.dropout_rate_ph: 0.0})

            # the test step contain evalate the indices and scores of crf
            # may analogy to the highest score pagerank node and score in
            # tf.contrib.crf.viterbi_decode
            # suggest only used in test time

            logits, sequence_actual_length, transaction_params =\
            self.sess.run([self.logits, self.sequence_actual_length, self.transition_params], feed_dict = feed_dict)

            # test the seq in seq_len slice(zip)
            for logit, seq_len in zip(logits, sequence_actual_length):
                logiit_actual = logit[:seq_len]

                # return index and cor score
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    logiit_actual,
                    transaction_params
                )
                viterbi_sequences.append(viterbi_sequence)


        print("the total len of sentence : %d" % data_count)
        return viterbi_sequences


    def evaluate(self, data_dict):
        data_count = data_dict["label"].shape[0]
        nb_eval = int(math.ceil(data_count / float(self.batch_size)))
        eval_loss = 0.0

        for i in range(nb_eval):
            feed_dict = dict()
            batch_indices = np.arange(i * self.batch_size, (i + 1) * self.batch_size)\
            if (i + 1) * self.batch_size <= data_count else np.arange(i * self.batch_size, data_count)

            for feature_name in self.feature_names:
                batch_data = data_dict[feature_name][batch_indices]
                # input data
                item = {self.input_feature_ph_dict[feature_name]: batch_data}
                feed_dict.update(item)

                # drop out prob set to 0 for eval
                item = {self.weight_dropout_ph_dict[feature_name]: 0.0}
                feed_dict.update(item)

            feed_dict.update({self.dropout_rate_ph: 0.0})

            # labels feed
            batch_label = data_dict["label"][batch_indices]
            feed_dict.update({self.input_label_ph: batch_label})

            loss = self.sess.run(self.loss, feed_dict = feed_dict)
            eval_loss += loss
        eval_loss /= float(nb_eval)
        return eval_loss


    def split_train_dev(self, data_dict, dev_size = 0.2):
        data_train_dict, data_dev_dict = dict(), dict()
        for name in data_dict.keys():
            boundary = int((1.0 - dev_size) * data_dict[name].shape[0])
            data_train_dict[name] = data_dict[name][:boundary]
            data_dev_dict[name] = data_dict[name][boundary:]

        return data_train_dict, data_dev_dict


    def compute_loss(self):

        # one_hot_encoding transform every element to one_hot tensor
        # maintain the ori order
        if not self.use_crf:
            labels = tf.reshape(
                tf.contrib.layers.one_hot_encoding(
                    tf.reshape(self.input_label_ph, [-1]), num_classes = self.nb_classes
                ),
                shape = [-1, self.sequence_length, self.nb_classes]
            )

            cross_entrogy = -1 * tf.reduce_sum(labels * tf.log(self.logits), axis=2)

            # the mask shape [batch_size, self.sentence_length] and all elements should be one
            # following will use length of true sequence of sentence to adjust the weight
            # of entrogy the longer the smaller
            mask = tf.sign(tf.reduce_max(tf.abs(labels), axis = 2))
            cross_entrogy_masked = tf.reduce_sum(
                cross_entrogy * mask, axis=1
            ) / tf.cast(self.sequence_actual_length, tf.float32)

            return tf.reduce_mean(cross_entrogy_masked)
        else:
            log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.input_label_ph, self.sequence_actual_length
            )

            return tf.reduce_mean(-1 * log_likelihood)































