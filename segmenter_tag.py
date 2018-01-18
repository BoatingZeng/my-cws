"""
tag 接口，专门做tag
"""

import argparse
import os
import reader
import toolbox
import codecs
import tensorflow as tf
from time import time
from model import Model
import pickle


def tag(path=None, model='trained_model'):
    assert path is not None
    assert model is not None
    assert os.path.isfile(path + '/chars.txt')

    if not os.path.isfile(path + '/' + model + '_model') or not os.path.isfile(
            path + '/' + model + '_weights.index'):
        raise Exception('No model file or weights file under the name of ' + model + '.')

    fin = open(path + '/' + model + '_model', 'rb')

    weight_path = path + '/' + model

    param_dic = pickle.load(fin)
    fin.close()

    nums_chars = param_dic['nums_chars']
    nums_tags = param_dic['nums_tags']
    crf = param_dic['crf']
    emb_dim = param_dic['emb_dim']
    gru = param_dic['gru']
    rnn_dim = param_dic['rnn_dim']
    rnn_num = param_dic['rnn_num']
    drop_out = param_dic['drop_out']
    buckets_char = param_dic['buckets_char']
    nums_ngrams = param_dic['ngram']
    is_space = param_dic['is_space']
    sent_seg = param_dic['sent_seg']
    emb_path = param_dic['emb_path']
    tag_scheme = param_dic['tag_scheme']

    ngram = 1

    char2idx, unk_chars, idx2char, tag2idx, idx2tag = toolbox.get_dicts(path, tag_scheme, crf)

    # trans_dict没有用到，但是暂时不方便删除
    trans_dict = {}

    new_chars = None

    test_x, test_y, raw_x, test_y_gold = None, None, None, None

    max_step = None

    raw_file = None
