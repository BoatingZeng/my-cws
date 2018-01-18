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

        self.char2idx, self.unk_chars, self.idx2char, self.tag2idx, self.idx2tag = toolbox.get_dicts(path, self.tag_scheme, self.crf)
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
            # TODO 先不考要改变embedding的大小
            # model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx)

            init = tf.global_variables_initializer()

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

        else:
            self.sess = [main_sess, None]

        with tf.device(self.gpu_config):
            print('Loading weights....')
            main_sess.run(init)
            self.model.run_updates(main_sess, weight_path + '_weights')

    def tag(self, lines, output_path):
        """

        :param lines: 字符串数组，相当于文件里的一行一行
        :return:
        """

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
        print('Raw setences: %d instances.' % len(raw_x[0]))
        max_step = self.sent_limit

        for k in range(len(raw_x)):
            raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

        with tf.device(self.gpu_config):
            self.model.tag(raw_x, lines, self.idx2tag, idx2char, unk_chars, self.trans_dict, self.sess, transducer=None,
                      outpath=output_path, batch_size=self.tag_batch)







def tag(path=None, model='trained_model', raw=None, output_path=None, segment_large=False, sent_limit=300, gpu=0,
        tag_batch=500, only_tokenised=False, large_size=10000):
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

    assert raw is not None

    raw_file = raw
    new_chars = toolbox.get_new_chars(raw_file, char2idx)

    if emb_path is not None:
        valid_chars = toolbox.get_valid_chars(new_chars, emb_path)
    else:
        valid_chars = None

    char2idx, idx2char, unk_chars = toolbox.update_char_dict(char2idx, new_chars, unk_chars, valid_chars)

    if not segment_large:
        raw_x, raw_len = toolbox.get_input_vec_raw(None, raw_file, char2idx, limit=sent_limit + 100)
        print('Raw setences: %d instances.' % len(raw_x[0]))
        max_step = raw_len
    else:
        max_step = sent_limit

    if not segment_large:
        for k in range(len(raw_x)):
            raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config = "/gpu:" + str(gpu)

    t = time()

    initializer = tf.contrib.layers.xavier_initializer()

    print('Initialization....')
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger") as scope:
            model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=[max_step], counts=[200],
                          crf=crf, ngram=nums_ngrams, batch_size=tag_batch)

            model.main_graph(trained_model=None, scope=scope, emb_dim=emb_dim, gru=gru,
                             rnn_dim=rnn_dim, rnn_num=rnn_num, drop_out=drop_out)

        model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx)

        init = tf.global_variables_initializer()

        print('Done. Time consumed: %d seconds' % int(time() - t))
    main_graph.finalize()
    idx = None

    main_sess = tf.Session(config=config, graph=main_graph)

    if crf:
        decode_graph = tf.Graph()

        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess, None]

    with tf.device(gpu_config):
        ens_model = None
        print('Loading weights....')
        main_sess.run(init)
        model.run_updates(main_sess, weight_path + '_weights')

        if not segment_large:
            raw_sents = []
            for line in codecs.open(raw_file, 'rb', encoding='utf-8'):
                line = line.strip()
                if len(line) > 0:
                    raw_sents.append(line)
            model.tag(raw_x, raw_sents, idx2tag, idx2char, unk_chars, trans_dict, sess, transducer=None,
                      outpath=output_path, batch_size=tag_batch, seg_large=segment_large)

        else:
            count = 0
            c_line = 0
            l_writer = codecs.open(output_path, 'w', encoding='utf-8')
            out = []
            with codecs.open(raw_file, 'r', encoding='utf-8') as l_file:
                lines = []
                for line in l_file:
                    line = line.strip()
                    if len(line) > 0:
                        lines.append(line)
                    else:
                        c_line += 1
                    if c_line >= large_size:
                        count += len(lines)
                        c_line = 0
                        print(count)
                        raw_x, _ = toolbox.get_input_vec_raw(None, None, char2idx, lines=lines, limit=sent_limit)

                        for k in range(len(raw_x)):
                            raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

                        predition, multi = model.tag(raw_x, lines, idx2tag, idx2char, unk_chars, trans_dict, sess,
                                                     transducer=None,
                                                     outpath=output_path, batch_size=tag_batch,
                                                     seg_large=segment_large)

                        if only_tokenised:
                            for l_out in predition:
                                if len(l_out.strip()) > 0:
                                    l_writer.write(l_out + '\n')
                        else:
                            for tagged_t, multi_t in zip(predition, multi):
                                if len(tagged_t.strip()) > 0:
                                    l_writer.write('#sent_tok: ' + tagged_t + '\n')
                                    idx = 1
                                    tgs = multi_t.split('  ')
                                    pl = ''
                                    for _ in range(8):
                                        pl += '\t' + '_'
                                    for tg in tgs:
                                        if '!#!' in tg:
                                            segs = tg.split('!#!')
                                            l_writer.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' + segs[
                                                0] + pl + '\n')
                                        else:
                                            l_writer.write(str(idx) + '\t' + tg + pl + '\n')
                                            idx += 1
                                    l_writer.write('\n')
                        lines = []
                if len(lines) > 0:

                    raw_x, _ = toolbox.get_input_vec_raw(None, None, char2idx, lines=lines, limit=sent_limit)

                    for k in range(len(raw_x)):
                        raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

                    prediction, multi = model.tag(raw_x, lines, idx2tag, idx2char, unk_chars, trans_dict, sess,
                                                  transducer=None, outpath=output_path, batch_size=tag_batch,
                                                  seg_large=segment_large)

                    if only_tokenised:
                        for l_out in prediction:
                            if len(l_out.strip()) > 0:
                                l_writer.write(l_out + '\n')
                    else:
                        for tagged_t, multi_t in zip(prediction, multi):
                            if len(tagged_t.strip()) > 0:
                                l_writer.write('#sent_tok: ' + tagged_t + '\n')
                                idx = 1
                                tgs = multi_t.split('  ')
                                pl = ''
                                for _ in range(8):
                                    pl += '\t' + '_'
                                for tg in tgs:
                                    if '!#!' in tg:
                                        segs = tg.split('!#!')
                                        l_writer.write(
                                            str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' + segs[0] + pl + '\n')
                                    else:
                                        l_writer.write(str(idx) + '\t' + tg + pl + '\n')
                                        idx += 1
                                l_writer.write('\n')
            l_writer.close()


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
    tag(path='./data/prechars_large', raw='./data/pku_raw.txt', output_path='./data/pku_seg.txt')