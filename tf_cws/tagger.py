"""
tag 接口，专门做tag
"""

import os
from . import toolbox
import tensorflow as tf
from time import time
from .model import Model
import pickle
import re


class Tagger(object):
    # 关于sent_limit的说明：
    # 短于这个参数的句子会补空白，空白太多会拖慢速度
    # 长于这个参数的句子，会被截断处理，不过输出结果还是合并的，正常的句子。截断对句子的标注有一定影响。
    # 所以sent_limit这个参数，设置成句子平均长度多一点就好了，所以默认20。
    # 由于toolbox里chop函数的截断策略，如果sen_limit小于等于10，会出错，所以干脆设置sent_limit最低值为20
    def __init__(self, path, model='trained_model', sent_limit=20, gpu=0, tag_batch=500):
        assert path is not None
        assert model is not None
        assert os.path.isfile(path + '/chars.txt')
        assert sent_limit >= 20

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
        config.gpu_options.allow_growth = True
        self.gpu_config = "/gpu:" + str(gpu)

        t = time()

        initializer = tf.contrib.layers.xavier_initializer()

        print('Initialization....')
        main_graph = tf.Graph()
        with main_graph.as_default():
            with tf.device(self.gpu_config):
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
                with tf.device(self.gpu_config):
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

    def tag(self, lines, isTokenize=False):
        """

        :param lines: 字符串数组，相当于文件里的一行一行
        :param isTokenize：是否要Tokenize，参考jieba。if True，会额外返回每个句子的tokenize列表。
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

        prediction_out = self.model.tag(raw_x, lines, self.idx2tag, idx2char, unk_chars, self.trans_dict, self.sess, transducer=None, batch_size=self.tag_batch)
        # TODO 在这里对英文和数字做特殊处理
        re_skip = re.compile('((?:[a-zA-Z0-9.%_\-/]+\s?)+)')
        seg_out = []
        token_out = []
        for seg, raw in zip(prediction_out, lines):
            seg_sp = re_skip.split(seg)
            raw_sp = re_skip.split(raw)
            assert len(seg_sp) == len(raw_sp)
            tmp = []
            for s, r in zip(seg_sp, raw_sp):
                if re_skip.match(s) is None:
                    tmp.append(s.strip())
                else:
                    tmp.append(r.strip())
            seg_line = ' '.join(tmp)
            seg_out.append(seg_line)

            if isTokenize:
                tmp_list = seg_line.split()
                token_line = []  # 元素：('这是', 0, 2)
                total_len = 0
                for e in tmp_list:
                    next_len = total_len + len(e)
                    token_line.append((e, total_len, next_len))
                    total_len = next_len
                token_out.append(token_line)

        if isTokenize:
            return seg_out, token_out
        else:
            return seg_out


def get_new_chars(lines, char2idx):
    new_chars = set()
    for line in lines:
        line = line.strip()
        for ch in line:
            if ch not in char2idx:
                new_chars.add(ch)
    return new_chars
