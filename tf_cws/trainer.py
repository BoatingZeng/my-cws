# -*- coding: utf-8 -*-
"""
@author: Boating Zeng
修改自 Yan Shao(yan.shao@lingfil.uu.se), https://github.com/yanshao9798/segmenter
"""
import argparse
import os
from . import reader
from . import toolbox
import codecs
import tensorflow as tf
from time import time
from .model import Model
import pickle


parser = argparse.ArgumentParser(description='A Universal Tokeniser. Written by Y. Shao, Uppsala University')
parser.add_argument('action', default='tag', choices=['train', 'test', 'tag'], help='train, test or tag')

parser.add_argument('-p', '--path', default=None, help='Path of the workstation')

parser.add_argument('-t', '--train', default=None, help='File for training')
parser.add_argument('-d', '--dev', default=None, help='File for validation')
parser.add_argument('-e', '--test', default=None, help='File for evaluation')
parser.add_argument('-r', '--raw', default=None, help='Raw file for tagging')

parser.add_argument('-m', '--model', default='trained_model', help='Name of the trained model')
parser.add_argument('-crf', '--crf', default=1, type=int, help='Using CRF interface')

parser.add_argument('-bt', '--bucket_size', default=50, type=int, help='Bucket size')
parser.add_argument('-sl', '--sent_limit', default=300, type=int, help='Long sentences will be chopped')

parser.add_argument('-tg', '--tags', default='BIES', help='Boundary Tagging, default is BIES')

parser.add_argument('-ed', '--embeddings_dimension', default=50, type=int, help='Dimension of the embeddings')
parser.add_argument('-emb', '--embeddings', default=None, help='Path and name of pre-trained char embeddings')

parser.add_argument('-ng', '--ngram', default=1, type=int, help='Using ngrams')

parser.add_argument('-gru', '--gru', default=False, help='Use GRU as the recurrent cell', action='store_true')
parser.add_argument('-rnn', '--rnn_cell_dimension', default=200, type=int, help='Dimension of the RNN cells')
parser.add_argument('-layer', '--rnn_layer_number', default=1, type=int, help='Numbers of the RNN layers')

parser.add_argument('-dr', '--dropout_rate', default=0.5, type=float, help='Dropout rate.这个参数是丢弃的概率，不是保留的概率。')

parser.add_argument('-iter', '--epochs', default=30, type=int, help='Numbers of epochs')

parser.add_argument('-op', '--optimizer', default='adagrad', help='Optimizer。选项：sgd, adagrad, adam, adadelta。')
parser.add_argument('-lr', '--learning_rate', default=0.2, type=float, help='Initial learning rate')
parser.add_argument('-ld', '--decay_rate', default=0.05, type=float, help='Learning rate decay')
parser.add_argument('-mt', '--momentum', default=None, type=float, help='Momentum')

parser.add_argument('-cp', '--clipping', default=False, action='store_true', help='Apply Gradient Clipping')

parser.add_argument("-tb","--train_batch", help="Training batch size", default=10, type=int)
parser.add_argument("-eb","--test_batch", help="Testing batch size", default=500, type=int)
parser.add_argument("-rb","--tag_batch", help="Tagging batch size", default=500, type=int)

parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

parser.add_argument('-opth', '--output_path', default=None, help='Output path')

parser.add_argument('-sgl', '--segment_large', default=False, help='Segment (very) large file', action='store_true')

parser.add_argument('-lgs', '--large_size', default=10000, type=int, help='Segment (very) large file')

parser.add_argument('-ot', '--only_tokenised', default=False, help='Only output the tokenised file when segment (very) large file',
                    action='store_true')

parser.add_argument('-ts', '--train_size', default=-1, type=int, help='No. of sentences used for training')

parser.add_argument('-rs', '--reset', default=False, help='Delete and re-initialise the intermediate files', action='store_true')

parser.add_argument('-sb', '--segmentation_bias', default=-1, type=float, help='Add segmentation bias to under(over)-splitting')

parser.add_argument('-ur', '--unk_rule', default=2, type=int, help='字典里，低于这个频数的字当作生僻字')

args = parser.parse_args()

if args.action == 'train':
    assert args.path is not None
    path = args.path
    train_file = args.train
    dev_file = args.dev
    model_file = args.model
    print('Reading data......')
    if args.reset or not os.path.isfile(path + '/raw_train.txt') or not os.path.isfile(path + '/raw_dev.txt'):
        # 用分好词的train和dev文件生成没有分词的raw文件
        if dev_file is None:
            reader.get_raw(path, train_file, '/raw_train.txt', is_dev=False)
        else:
            reader.get_raw(path, train_file, '/raw_train.txt')
            reader.get_raw(path, dev_file, '/raw_dev.txt')

    if args.reset or not os.path.isfile(path + '/tag_train.txt') or not os.path.isfile(path + '/tag_dev.txt') or \
            not os.path.isfile(path + '/tag_dev_gold.txt'):
        # 生成打了tag的文件，就是给没有分词的文件标上BIES
        if dev_file is None:
            raws_train = reader.raw(path + '/raw_train.txt')
            raws_dev = reader.raw(path + '/raw_dev.txt')
            sents_train, sents_dev = reader.gold(path + '/' + train_file, False)
        else:
            raws_train = reader.raw(path + '/raw_train.txt')
            sents_train = reader.gold(path + '/' + train_file)

            raws_dev = reader.raw(path + '/raw_dev.txt')
            sents_dev = reader.gold(path + '/' + dev_file)

        toolbox.raw2tags(raws_train, sents_train, path, 'tag_train.txt', reset=args.reset, tag_scheme=args.tags)
        toolbox.raw2tags(raws_dev, sents_dev, path, 'tag_dev.txt', gold_path='tag_dev_gold.txt', tag_scheme=args.tags)

    # if args.reset or not os.path.isfile(path + '/chars.txt'):
    #     toolbox.get_chars(path, ['raw_train.txt', 'raw_dev.txt'])

    '''
        unk_chars储存生僻字，这里把只出现一次的字作为生僻字了
    '''
    char2idx, unk_chars, idx2char, tag2idx, idx2tag = toolbox.get_dicts(path, args.tags, args.crf, args.unk_rule)

    #trans_dict没有用到，但是暂时不方便删除
    trans_dict = {}

    if args.embeddings is not None:
        print('Reading embeddings...')
        short_emb = args.embeddings[args.embeddings.index('/') + 1: args.embeddings.index('.')]
        if args.reset or not os.path.isfile(path + '/' + short_emb + '_sub.txt'):
            toolbox.get_sample_embedding(path, args.embeddings, char2idx)
        emb_dim, emb, valid_chars = toolbox.read_sample_embedding(path, short_emb, char2idx)
        for vch in valid_chars:
            if vch in unk_chars:
                unk_chars.remove(vch)
    else:
        emb_dim = args.embeddings_dimension
        emb = None

    # 如果语料里出现不在字典里的字，就把它指向<UNK>
    train_x, train_y, max_len_train = toolbox.get_input_vec(path, 'tag_train.txt', char2idx, tag2idx,
                                                            limit=args.sent_limit, train_size=args.train_size)

    dev_x, max_len_dev = toolbox.get_input_vec_raw(path, 'raw_dev.txt', char2idx, limit=args.sent_limit)

    print('Training set: %d instances; Dev set: %d instances.' % (len(train_x[0]), len(dev_x[0])))

    nums_grams = None

    max_len = max(max_len_train, max_len_dev)

    b_train_x, b_train_y = toolbox.buckets(train_x, train_y, size=args.bucket_size)
    b_train_x, b_train_y, b_lens, b_count = toolbox.pad_bucket(b_train_x, b_train_y, max_len)

    b_dev_x = [toolbox.pad_zeros(dev_x_i, max_len) for dev_x_i in dev_x]

    b_dev_y_gold = [line.strip() for line in codecs.open(path + '/tag_dev_gold.txt', 'r', encoding='utf-8')]

    nums_tag = len(tag2idx)

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config = "/gpu:" + str(args.gpu)

    initializer = tf.contrib.layers.xavier_initializer()

    print('Initialization....')
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.device(gpu_config):
            with tf.variable_scope("tagger") as scope:
                model = Model(nums_chars=len(char2idx) + 2, nums_tags=nums_tag, buckets_char=b_lens, counts=b_count,
                              crf=args.crf, ngram=nums_grams, batch_size=args.train_batch,
                              emb_path=args.embeddings, tag_scheme=args.tags)

                model.main_graph(trained_model=path + '/' + model_file + '_model', scope=scope,
                                 emb_dim=emb_dim, gru=args.gru, rnn_dim=args.rnn_cell_dimension,
                                 rnn_num=args.rnn_layer_number, drop_out=args.dropout_rate, emb=emb, unk_rule=args.unk_rule)
                t = time()

            model.config(optimizer=args.optimizer, decay=args.decay_rate, lr_v=args.learning_rate,
                         momentum=args.momentum, clipping=args.clipping)

            init = tf.global_variables_initializer()

            # 保存graph
            # writer = tf.summary.FileWriter('./data/graphs/train/main_graph', main_graph)
            # writer.close()

            print('Done. Time consumed: %d seconds' % int(time() - t))

    main_graph.finalize()
    main_sess = tf.Session(config=config, graph=main_graph)

    if args.crf > 0:
        decode_graph = tf.Graph()
        with decode_graph.as_default():
            with tf.device(gpu_config):
                model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        # 保存graph
        # writer = tf.summary.FileWriter('./data/graphs/train/decode_graph', decode_graph)
        # writer.close()
        # print('保存之后退出，不继续')
        # sys.exit()

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess, None]

    #with tf.device(gpu_config):

    main_sess.run(init)
    print('Initialisation...')
    print('Done. Time consumed: %d seconds' % int(time() - t))
    print('开始model.train函数')
    t = time()
    b_dev_raw = [line.strip() for line in codecs.open(path + '/raw_dev.txt', 'r', encoding='utf-8')]
    model.train(b_train_x, b_train_y, b_dev_x, b_dev_raw, b_dev_y_gold, idx2tag, idx2char, unk_chars, trans_dict,
                sess, args.epochs, path + '/' + model_file + '_weights', lr=args.learning_rate,
                decay=args.decay_rate, outpath=args.output_path)

else:

    assert args.path is not None
    assert args.model is not None
    path = args.path
    assert os.path.isfile(path + '/chars.txt')

    model_file = args.model

    if not os.path.isfile(path + '/' + model_file + '_model') or not os.path.isfile(
            path + '/' + model_file + '_weights.index'):
        raise Exception('No model file or weights file under the name of ' + model_file + '.')

    fin = open(path + '/' + model_file + '_model', 'rb')

    weight_path = path + '/' + model_file

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
    unk_rule = param_dic['unk_rule']

    if args.embeddings is not None:
        emb_path = args.embeddings

    ngram = 1

    char2idx, unk_chars, idx2char, tag2idx, idx2tag = toolbox.get_dicts(path, tag_scheme, crf, unk_rule)

    # trans_dict没有用到，但是暂时不方便删除
    trans_dict = {}

    new_chars = None

    test_x, test_y, raw_x, test_y_gold = None, None, None, None

    max_step = None

    raw_file = None

    if args.action == 'test':
        test_file = args.test
        assert test_file is not None

        reader.get_raw(path, test_file, 'raw_test.txt')

        raws_test = reader.raw(path + '/raw_test.txt')
        test_y_gold = reader.test_gold(path + '/' + test_file)

        new_chars = toolbox.get_new_chars(path + '/raw_test.txt', char2idx)

        if emb_path is not None:
            valid_chars = toolbox.get_valid_chars(new_chars + list(char2idx.keys()), emb_path)
        else:
            valid_chars = None

        char2idx, idx2char, unk_chars = toolbox.update_char_dict(char2idx, new_chars, unk_chars, valid_chars)

        test_x, max_len_test = toolbox.get_input_vec_raw(path, 'raw_test.txt', char2idx, limit=args.sent_limit + 100)

        max_step = max_len_test

        print('Test set: %d instances.' % len(test_x[0]))

        for k in range(len(test_x)):
            test_x[k] = toolbox.pad_zeros(test_x[k], max_step)

    elif args.action == 'tag':
        assert args.raw is not None

        raw_file = args.raw
        new_chars = toolbox.get_new_chars(raw_file, char2idx)

        if emb_path is not None:
            valid_chars = toolbox.get_valid_chars(new_chars, emb_path)
        else:
            valid_chars = None

        char2idx, idx2char, unk_chars = toolbox.update_char_dict(char2idx, new_chars, unk_chars, valid_chars)

        if not args.segment_large:
            raw_x, raw_len = toolbox.get_input_vec_raw(None, raw_file, char2idx, limit=args.sent_limit + 100)
            print('Raw setences: %d instances.' % len(raw_x[0]))
            max_step = raw_len
        else:
            max_step = args.sent_limit

        if not args.segment_large:
            for k in range(len(raw_x)):
                raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)


    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config = "/gpu:" + str(args.gpu)

    t = time()

    initializer = tf.contrib.layers.xavier_initializer()

    print('Initialization....')
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.device(gpu_config):
            with tf.variable_scope("tagger") as scope:
                model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=[max_step], counts=[200],
                              crf=crf, ngram=nums_ngrams, batch_size=args.tag_batch)

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
            with tf.device(gpu_config):
                model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess, None]


#with tf.device(gpu_config):
    ens_model = None
    print('Loading weights....')
    main_sess.run(init)
    model.run_updates(main_sess, weight_path + '_weights')

    if args.action == 'test':
        test_y_raw = [line.strip() for line in codecs.open(path + '/raw_test.txt', 'rb', encoding='utf-8')]
        model.test(test_x, test_y_raw, test_y_gold, idx2tag, idx2char, unk_chars, trans_dict, sess, transducer=None,
                   batch_size=args.test_batch, bias=args.segmentation_bias, outpath=args.output_path)

    if args.action == 'tag':
        if not args.segment_large:
            raw_sents = []
            for line in codecs.open(raw_file, 'rb', encoding='utf-8'):
                line = line.strip()
                if len(line) > 0:
                    raw_sents.append(line)
            model.tag(raw_x, raw_sents, idx2tag, idx2char, unk_chars, trans_dict, sess, transducer=None,
                      outpath=args.output_path, batch_size=args.tag_batch, seg_large=args.segment_large)

        else:
            count = 0
            c_line = 0
            l_writer = codecs.open(args.output_path, 'w', encoding='utf-8')
            out = []
            with codecs.open(raw_file, 'r', encoding='utf-8') as l_file:
                lines = []
                for line in l_file:
                    line = line.strip()
                    if len(line) > 0:
                        lines.append(line)
                    else:
                        c_line += 1
                    if c_line >= args.large_size:
                        count += len(lines)
                        c_line = 0
                        print(count)
                        raw_x, _ = toolbox.get_input_vec_raw(None, None, char2idx, lines=lines, limit=args.sent_limit)

                        for k in range(len(raw_x)):
                            raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

                        predition, multi = model.tag(raw_x, lines, idx2tag, idx2char, unk_chars, trans_dict, sess, transducer=None,
                                                     outpath=args.output_path, batch_size=args.tag_batch, seg_large=args.segment_large)

                        if args.only_tokenised:
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
                                            l_writer.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' + segs[0] + pl + '\n')
                                        else:
                                            l_writer.write(str(idx) + '\t' + tg + pl + '\n')
                                            idx += 1
                                    l_writer.write('\n')
                        lines = []
                if len(lines) > 0:

                    raw_x, _ = toolbox.get_input_vec_raw(None, None, char2idx, lines=lines, limit=args.sent_limit)

                    for k in range(len(raw_x)):
                        raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

                    prediction, multi = model.tag(raw_x, lines, idx2tag, idx2char, unk_chars, trans_dict, sess,
                                                 transducer=None, outpath=args.output_path, batch_size=args.tag_batch, seg_large=args.segment_large)

                    if args.only_tokenised:
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
                                        l_writer.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' + segs[0] + pl + '\n')
                                    else:
                                        l_writer.write(str(idx) + '\t' + tg + pl + '\n')
                                        idx += 1
                                l_writer.write('\n')
            l_writer.close()
