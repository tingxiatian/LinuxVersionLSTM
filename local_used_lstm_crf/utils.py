#coding: utf-8
from __future__ import print_function
import os
import codecs
import pickle
import numpy as np
import tensorflow as tf

def uniform_tensor(shape, name, dtype = "float32"):
    return tf.random_uniform(shape=shape, minval=-1.0, maxval=1.0, dtype=tf.float32, name = name)

#this function will used to retrieve the sequence tensor non-negative length so work(tf.sign)
def get_sequence_actual_length(tensor):
    actual_length = tf.reduce_sum(tf.sign(tensor), axis = 1)
    return tf.cast(actual_length, tf.int32)

def zero_nil_slot(t, name = None):
    with tf.name_scope("zero_nil_slot"):
        s = tf.shape(t)[1]
        z = tf.zeros([1, s], dtype=tf.float32)

        # the slice method begin in [1, 0] and end with size [-1, -1]
        # the -1 slice to the tail
        # put zeros to the head of t
        return tf.concat(
            axis=0, name = name,
            values=[z, tf.slice(t, [1, 0], [-1, -1])]
        )

# shuffle list of matrix
def shuffle_matrix(*args, **kw):
    seed = kw["seed"] if "seed" in kw else 1337
    for arg in args:
        np.random.seed(seed)
        np.random.shuffle(arg)

# process token of counts and filter by counts
# serial the word index dict
def create_dictionary(token_dict, dic_path, start = 0, sort = False,
                      min_count = None, lower = False, overwrite = False):
    if os.path.exists(dic_path) and not overwrite:
        return 0

    voc = dict()
    if sort:
        token_list = sorted(token_dict.items(), key = lambda d: d[1], reverse=True)
        for i, item in enumerate(token_list):
            if min_count and item[1] < min_count:
                continue
            index = i + start
            key = item[0]
            voc[key] = index
    else:
        if min_count:
            items = sorted([item[0] for item in token_dict.items() if item[1] >= min_count])
        else:
            items = sorted([item[0] for item in token_dict.items()])
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            index = i + start
            voc[item] = index

    file = open(dic_path, "wb")
    pickle.dump(voc, file)
    file.close()
    return len(voc.keys())

# the non_word setting the word not contained in the voc dict
# the init_value may not be zero
# return the correspond index in the ori voc dict
def map_item2id(items, voc, max_len, non_word = 1, lower = False, init_value = 0, allow_error = False):
    assert type(non_word) == int
    arr = np.zeros((max_len,), dtype="int32") + init_value
    min_range = min(max_len, len(items))
    for i in range(min_range):
        item = items[i] if not lower else items[i].lower()
        if allow_error:
            arr[i] = voc[item] if item in voc else non_word
        else:
            arr[i] = voc[item]
    return arr

# the return is the token_weight which is the dense representation of word(word embedding)
# and the un-register word
# wrt un-register word use random uniform vector to express in weight
def build_lookup_table(vec_dim, token2id_dict, token2vec_dict = None, token_vec_start = 1):
    unknow_token_count = 0
    token_voc_size = len(token2id_dict.keys()) + token_vec_start

    if token2vec_dict is None:
        token_weight = np.random.normal(size = (token_voc_size, vec_dim)).astype("float32")
        for i in range(token_vec_start):
            token_weight[i, :] = 0.
        return token_weight, 0

    token_weight = np.zeros(shape=(token_voc_size, vec_dim), dtype="float32")
    for token in token2id_dict:
        index = token2id_dict[token]
        if token in token2vec_dict:
            token_weight[index, :] = token2vec_dict[token]
        else:
            unknow_token_count += 1
            random_vec = np.random.uniform(-0.25, 0.25, size = (vec_dim,)).astype("float32")
            token_weight[index, :] = random_vec

    return token_weight, unknow_token_count

def embedding_txt2pkl(path_txt, path_pkl):
    print("convert txt to pickle...")
    from gensim.models.keyedvectors import KeyedVectors
    assert path_txt.endswith("txt")
    word_vectors = KeyedVectors.load_word2vec_format(path_txt, binary=False)
    word_dict = dict()

    for word in word_vectors.vocab:
        word_dict[word] = word_vectors[word]

    with open(path_pkl, "wb") as f:
        pickle.dump(word_dict, f)

def load_embed_from_txt(path):
    file_r = codecs.open(path, "r", encoding="utf-8")
    line = file_r.readline()
    embedding = dict()
    voc_size, vec_dim = map(int, line.split(" "))
    while line:
        items = line.split(" ")
        item = items[0]
        vec = np.array(items[1:], dtype= "float32")
        embedding[item] = vec
        line = file_r.readline()

    return embedding, vec_dim



















if __name__ == "__main__":
    test_tensor = tf.Variable(initial_value=tf.random_uniform(shape=[3, 4]))

    t0 = tf.Variable(initial_value=tf.random_uniform(shape=[1, 2]))
    t1 = tf.Variable(initial_value=tf.random_uniform(shape=[1, 2]))
    t2 = tf.Variable(initial_value=tf.random_uniform(shape=[2, 2]))

    t3 = tf.Variable(initial_value=tf.random_uniform(minval=-10.0, maxval=-1.0, shape=[3, 4]))

    #t4 = tf.Variable(initial_value=tf.random_uniform())
    input_int_array = np.random.randint(low = 0, high=10, size = [5, 3])
    const_input_tensor = tf.constant(input_int_array)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("ori tensor :")
        print(sess.run(test_tensor))
        print(sess.run(get_sequence_actual_length(test_tensor)))

        print("t2 :")
        print(sess.run(t2))
        print("slice :")

        '''
        print(tf.slice(t2, [0], [1]))
        print(tf.slice(t2, [0, 1], [-1, -1]))    
        '''


        print("slot :")
        print(sess.run(zero_nil_slot(t2)))

        t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                         [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 6, 6]]])
        '''
        [[3, 3, 3], [4, 4, 4]]
        '''

        #print(sess.run(tf.slice(t, [1, 0, 0], [-1, -1, -1])))
        print("\nthe t3 shape :")
        print(sess.run(t3))
        print(sess.run(get_sequence_actual_length(t3)))

        print("\n" * 3)
        print("one hot encoding test :")
        print(sess.run(const_input_tensor))
        print("the batch size of input is 5, the dim is 3")
        encoding_tensor = tf.contrib.layers.one_hot_encoding(const_input_tensor, 10)
        print(sess.run(encoding_tensor))

        input_label_ph_init = np.random.randint(0, 10, size = [5, 3])
        input_label_ph = tf.constant(input_label_ph_init)

        print("the mask :")

        labels = tf.reshape(
            tf.contrib.layers.one_hot_encoding(
                tf.reshape(input_label_ph, [-1]), num_classes=10),
            shape=[-1,
            3, 10])



        print("labels :")
        print(sess.run(labels))

        labels_mask = tf.sign(tf.reduce_max(tf.abs(labels), axis=2))
        print("mask labels :")
        print(sess.run(labels_mask))




