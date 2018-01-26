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


class Tagger(object):

    def __init__(self, path, model='trained_model', sent_limit=300, gpu=0, tag_batch=500):
        assert path is not None
        assert model is not None
        assert os.path.isfile(path + '/chars.txt')

        if not os.path.isfile(path + '/' + model + '_model') or not os.path.isfile(
                path + '/' + model + '_weights.index'):
            raise Exception('No model file or weights file under the name of ' + model + '.')

        fin = open(path + '/' + model + '_model', 'rb')

        self.sent_limit = sent_limit
        self.gpu = gpu
        self.tag_batch = tag_batch
        weight_path = path + '/' + model

        param_dic = pickle.load(fin)
        fin.close()

        self.nums_chars = param_dic['nums_chars']
        self.nums_tags = param_dic['nums_tags']
        self.crf = param_dic['crf']
        self.emb_dim = param_dic['emb_dim']
        self.gru = param_dic['gru']
        self.rnn_dim = param_dic['rnn_dim']
        self.rnn_num = param_dic['rnn_num']
        self.drop_out = param_dic['drop_out']
        self.buckets_char = param_dic['buckets_char']
        self.nums_ngrams = param_dic['ngram']
        self.is_space = param_dic['is_space']
        self.sent_seg = param_dic['sent_seg']
        self.emb_path = param_dic['emb_path']
        self.tag_scheme = param_dic['tag_scheme']
        self.unk_rule = param_dic['unk_rule']

        self.char2idx, self.unk_chars, self.idx2char, self.tag2idx, self.idx2tag = toolbox.get_dicts(path, self.tag_scheme, self.crf, self.unk_rule)
        self.trans_dict = {}

        config = tf.ConfigProto(allow_soft_placement=True)
        self.gpu_config = "/gpu:" + str(gpu)

        t = time()

        initializer = tf.contrib.layers.xavier_initializer()

        print('Initialization....')
        main_graph = tf.Graph()
        with main_graph.as_default():
            with tf.variable_scope("tagger") as scope:
                self.model = Model(nums_chars=self.nums_chars, nums_tags=self.nums_tags, buckets_char=[self.sent_limit], counts=[200],
                              crf=self.crf, ngram=self.nums_ngrams, batch_size=self.tag_batch)

                self.model.main_graph(trained_model=None, scope=scope, emb_dim=self.emb_dim, gru=self.gru,
                                 rnn_dim=self.rnn_dim, rnn_num=self.rnn_num, drop_out=self.drop_out)
            # TODO 先不要改变embedding的大小
            # model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx)

            init = tf.global_variables_initializer()

            # 保存graph
            # writer = tf.summary.FileWriter('./data/graphs/tag/main_graph', main_graph)
            # writer.close()

            print('Done. Time consumed: %d seconds' % int(time() - t))
        main_graph.finalize()
        idx = None

        main_sess = tf.Session(config=config, graph=main_graph)

        if self.crf:
            decode_graph = tf.Graph()

            with decode_graph.as_default():
                self.model.decode_graph()
            decode_graph.finalize()

            decode_sess = tf.Session(config=config, graph=decode_graph)

            self.sess = [main_sess, decode_sess]

            # 保存graph
            # writer = tf.summary.FileWriter('./data/graphs/tag/decode_graph', decode_graph)
            # writer.close()

        else:
            self.sess = [main_sess, None]

        with tf.device(self.gpu_config):
            print('Loading weights....')
            main_sess.run(init)
            self.model.run_updates(main_sess, weight_path + '_weights')

    # 要释放一些资源
    def __del__(self):
        # 关闭session
        for s in self.sess:
            s.close()

    def tag(self, lines):
        """

        :param lines: 字符串数组，相当于文件里的一行一行
        :return:
        """
        lines = [line.strip() for line in lines]
        # 这些是每次tag时的用的
        new_chars = get_new_chars(lines, self.char2idx)
        if self.emb_path is not None:
            valid_chars = toolbox.get_valid_chars(new_chars, self.emb_path)
        else:
            valid_chars = None

        char2idx, idx2char, unk_chars = toolbox.update_char_dict(self.char2idx, new_chars, self.unk_chars, valid_chars)

        # 因为build graph的时候要用到max_step，但是每次tag的时候build graph并不是所期望的
        # 所以max_step设置为初始化时的sent_limit
        raw_x, raw_len = toolbox.get_input_vec_raw(None, None, char2idx, lines, limit=self.sent_limit)
        # print('Raw setences: %d instances.' % len(raw_x[0]))
        max_step = self.sent_limit

        for k in range(len(raw_x)):
            raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

        with tf.device(self.gpu_config):
            prediction_out, multi_out = self.model.tag(raw_x, lines, self.idx2tag, idx2char, unk_chars, self.trans_dict, self.sess, transducer=None, batch_size=self.tag_batch)
        return prediction_out


def get_new_chars(lines, char2idx):
    new_chars = set()
    for line in lines:
        line = line.strip()
        for ch in line:
            if ch not in char2idx:
                new_chars.add(ch)
    return new_chars


if __name__ == '__main__':
    print('测试')
    tagger = Tagger('./data/prechars_large')

    # lines = codecs.open('./data/pku_raw.txt', 'rb', encoding='utf-8')
    # t = time()
    # tagger.tag(lines)
    # print(time()-t)
