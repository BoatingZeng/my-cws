# -*- coding: utf-8 -*-
import tensorflow as tf
from layers import EmbeddingLayer, BiLSTM, HiddenLayer, TimeDistributed, DropoutLayer, Convolution, Maxpooling, Forward
from time import time
import losses
import toolbox
import batch as Batch
import random
import pickle as pickle
import codecs
import evaluation
import os


class Model(object):

    def __init__(self, nums_chars, nums_tags, buckets_char, counts=None, batch_size=10, crf=1, ngram=None, sent_seg=False,
                 is_space=False, emb_path=None, tag_scheme='BIES'):
        self.nums_chars = nums_chars
        self.nums_tags = nums_tags
        self.buckets_char = buckets_char
        self.counts = counts
        self.crf = crf
        self.ngram = ngram
        self.emb_path = emb_path
        self.emb_layer = None
        self.tag_scheme = tag_scheme
        self.gram_layers = []
        self.batch_size = batch_size
        self.l_rate = None
        self.decay = None
        self.train_step = None
        self.saver = None
        self.decode_holders = None
        self.scores = None
        self.params = None
        self.pixels = None
        self.is_space = is_space
        self.sent_seg = sent_seg
        self.updates = []
        self.bucket_dit = {}
        self.input_v = []
        self.input_w = []
        self.input_p = None
        self.output = []
        self.output_ = []
        self.output_p = []

        if self.crf > 0:
            self.transition_char = tf.get_variable('transitions_char', [self.nums_tags + 1, self.nums_tags + 1])
        else:
            self.transition_char = None

        while len(self.buckets_char) > len(self.counts):
            self.counts.append(1)

        self.real_batches = toolbox.get_real_batch(self.counts, self.batch_size)

    def main_graph(self, trained_model, scope, emb_dim, gru, rnn_dim, rnn_num, drop_out=0.5, emb=None, unk_rule=2):
        if trained_model is not None:
            param_dic = {'nums_chars': self.nums_chars, 'nums_tags': self.nums_tags, 'crf': self.crf, 'emb_dim': emb_dim,
                         'gru': gru, 'rnn_dim': rnn_dim, 'rnn_num': rnn_num, 'drop_out': drop_out, 'buckets_char': self.buckets_char,
                         'ngram': self.ngram, 'is_space': self.is_space, 'sent_seg': self.sent_seg, 'emb_path': self.emb_path,
                         'tag_scheme': self.tag_scheme, 'unk_rule': unk_rule}
            #print param_dic
            # 如果已经有model的基本信息，就不要再导出了
            if not os.path.isfile(trained_model):
                f_model = open(trained_model, 'wb')
                pickle.dump(param_dic, f_model)
                f_model.close()

        # define shared weights and variables

        dr = tf.placeholder(tf.float32, [], name='drop_out_holder')
        self.drop_out = dr
        self.drop_out_v = drop_out

        self.emb_layer = EmbeddingLayer(self.nums_chars + 20, emb_dim, weights=emb, name='emb_layer')

        if self.ngram is not None:
            ng_embs = [None for _ in range(len(self.ngram))]
            for i, n_gram in enumerate(self.ngram):
                self.gram_layers.append(EmbeddingLayer(n_gram + 5000 * (i + 2), emb_dim, weights=ng_embs[i], name= str(i + 2) + 'gram_layer'))

        with tf.variable_scope('BiRNN'):

            if gru:
                fw_rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
                bw_rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
            else:
                fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_dim, state_is_tuple=True)
                bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_dim, state_is_tuple=True)

            if rnn_num > 1:
                fw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([fw_rnn_cell]*rnn_num, state_is_tuple=True)
                bw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([bw_rnn_cell]*rnn_num, state_is_tuple=True)

        output_wrapper = TimeDistributed(HiddenLayer(rnn_dim * 2, self.nums_tags, activation='linear', name='hidden'), name='wrapper')

        #define model for each bucket
        for idx, bucket in enumerate(self.buckets_char):
            if idx == 1:
                scope.reuse_variables()
            t1 = time()

            input_v = tf.placeholder(tf.int32, [None, bucket], name='input_' + str(bucket))

            self.input_v.append([input_v])

            emb_set = []

            word_out = self.emb_layer(input_v)
            emb_set.append(word_out)

            if self.ngram is not None:
                for i in range(len(self.ngram)):
                    input_g = tf.placeholder(tf.int32, [None, bucket], name='input_g' + str(i) + str(bucket))
                    self.input_v[-1].append(input_g)
                    gram_out = self.gram_layers[i](input_g)
                    emb_set.append(gram_out)

            if len(emb_set) > 1:
                emb_out = tf.concat(axis=2, values=emb_set)

            else:
                emb_out = emb_set[0]

            emb_out = DropoutLayer(dr)(emb_out)

            rnn_out = BiLSTM(rnn_dim, nums_layers=rnn_num, fw_cell=fw_rnn_cell, bw_cell=bw_rnn_cell, p=dr, name='BiLSTM' + str(bucket), scope='BiRNN')(emb_out, input_v)

            output = output_wrapper(rnn_out)

            self.output.append([output])

            self.output_.append([tf.placeholder(tf.int32, [None, bucket], name='tags' + str(bucket))])
            self.bucket_dit[bucket] = idx

            print('Bucket %d, %f seconds' % (idx + 1, time() - t1))

        assert len(self.input_v) == len(self.output) and len(self.output) == len(self.output_) and len(self.output) == len(self.counts)

        self.params = tf.trainable_variables()

        self.saver = tf.train.Saver()

    def config(self, optimizer, decay, lr_v=None, momentum=None, clipping=False, max_gradient_norm=5.0):

        self.decay = decay
        print('Training preparation...')

        print('Defining loss...')
        loss = []
        if self.crf > 0:
            loss_function = losses.crf_loss
            for i in range(len(self.input_v)):
                bucket_loss = losses.loss_wrapper(self.output[i], self.output_[i], loss_function,
                                                  transitions=[self.transition_char], nums_tags=[self.nums_tags], batch_size=self.real_batches[i])
                loss.append(bucket_loss)
        else:
            loss_function = losses.sparse_cross_entropy
            for output, output_ in zip(self.output, self.output_):
                bucket_loss = losses.loss_wrapper(output, output_, loss_function)
                loss.append(bucket_loss)

        l_rate = tf.placeholder(tf.float32, [], name='learning_rate_holder')
        self.l_rate = l_rate

        if optimizer == 'sgd':
            if momentum is None:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=momentum)
        elif optimizer == 'adagrad':
            assert lr_v is not None
            optimizer = tf.train.AdagradOptimizer(learning_rate=l_rate)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer()
        elif optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)  # 根据文档，试试设置固定的初始lr
        else:
            raise Exception('optimiser error')

        self.train_step = []

        print('Computing gradients...')

        for idx, l in enumerate(loss):
            t2 = time()
            if clipping:
                gradients = tf.gradients(l, self.params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                train_step = optimizer.apply_gradients(list(zip(clipped_gradients, self.params)))
            else:
                train_step = optimizer.minimize(l)
            print('Bucket %d, %f seconds' % (idx + 1, time() - t2))
            self.train_step.append(train_step)

    def decode_graph(self):
        self.decode_holders = []
        self.scores = []
        for bucket in self.buckets_char:
            decode_holders = []
            scores = []
            nt = self.nums_tags
            ob = tf.placeholder(tf.float32, [None, bucket, nt])
            trans = tf.placeholder(tf.float32, [nt + 1, nt + 1])
            nums_steps = ob.get_shape().as_list()[1]
            length = tf.placeholder(tf.int32, [None])
            b_size = tf.placeholder(tf.int32, [])
            small = -1000
            class_pad = tf.stack(small * tf.ones([b_size, nums_steps, 1]))
            observations = tf.concat(axis=2, values=[ob, class_pad])
            b_vec = tf.tile(([small] * nt + [0]), [b_size])
            b_vec = tf.cast(b_vec, tf.float32)
            b_vec = tf.reshape(b_vec, [b_size, 1, -1])
            observations = tf.concat(axis=1, values=[b_vec, observations])
            transitions = tf.reshape(tf.tile(trans, [b_size, 1]), [b_size, nt + 1, nt + 1])
            observations = tf.reshape(observations, [-1, nums_steps + 1, nt + 1, 1])
            observations = tf.transpose(observations, [1, 0, 2, 3])
            previous = observations[0, :, :, :]
            max_scores = []
            max_scores_pre = []
            alphas = [previous]
            for t in range(1, nums_steps + 1):
                previous = tf.reshape(previous, [-1, nt + 1, 1])
                current = tf.reshape(observations[t, :, :, :], [-1, 1, nt + 1])
                alpha_t = previous + current + transitions
                max_scores.append(tf.reduce_max(alpha_t, axis=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
                alpha_t = tf.reshape(Forward.log_sum_exp(alpha_t, axis=1), [-1, nt + 1, 1])
                alphas.append(alpha_t)
                previous = alpha_t
            max_scores = tf.stack(max_scores, axis=1)
            max_scores_pre = tf.stack(max_scores_pre, axis=1)
            decode_holders.append([ob, trans, length, b_size])
            scores.append((max_scores, max_scores_pre))
            self.decode_holders.append(decode_holders)
            self.scores.append(scores)

    def define_updates(self, new_chars, emb_path, char2idx):
        self.nums_chars += len(new_chars)

        if emb_path is not None:
            old_emb_weights = self.emb_layer.embeddings
            emb_dim = old_emb_weights.get_shape().as_list()[1]
            emb_len = old_emb_weights.get_shape().as_list()[0]
            new_emb = tf.stack(toolbox.get_new_embeddings(new_chars, emb_dim, emb_path))
            n_emb_sh = new_emb.get_shape().as_list()
            if len(n_emb_sh) > 1:
                new_emb_weights = tf.concat(axis=0, values=[old_emb_weights[:len(char2idx) - len(new_chars)], new_emb, old_emb_weights[len(char2idx):]])
                if new_emb_weights.get_shape().as_list()[0] > emb_len:
                    new_emb_weights = new_emb_weights[:emb_len]
                assign_op = old_emb_weights.assign(new_emb_weights)
                self.updates.append(assign_op)

    def run_updates(self, sess, weight_path, need_restore=True):
        if need_restore:
            self.saver.restore(sess, weight_path)
        for op in self.updates:
            sess.run(op)
        # 每次update后清空
        self.updates = []

        print('Loaded.')

    def define_transducer_dict(self, trans_str, char2idx, sess, transducer):
        indices = []
        for ch in trans_str:
            if ch == ' ':
                indices.append(3)
            elif ch in char2idx:
                indices.append(char2idx[ch])
            else:
                indices.append(char2idx['<UNK>'])
        indices += [2]
        out = transducer.tag([indices], char2idx, sess, batch_size=1)
        out = out[0].replace(' ', '  ')
        return out

    def train(self, t_x, t_y, v_x, v_y_raw, v_y_gold, idx2tag, idx2char, unk_chars, trans_dict, sess, epochs, trained_model,
              transducer=None, lr=0.05, decay=0.05, decay_step=1, sent_seg=False, outpath=None):

        # 在每次训练时读取前一次weights
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(trained_model))
        if ckpt and ckpt.model_checkpoint_path:
            print('读取前一次checkpoint')
            self.saver.restore(sess[0], ckpt.model_checkpoint_path)

        # 读取前一次保存的分数
        if os.path.isfile(trained_model + '_score'):
            print('读取前一次保存的分数')
            score_file = open(trained_model + '_score', 'rb')
            best_score = pickle.load(score_file)
            score_file.close()
        else:
            best_score = [0] * 6
        t_pre = time()
        lr_r = lr

        best_epoch = 0

        chars = toolbox.decode_chars(v_x[0], idx2char)

        # train集和dev集里的生僻字，指向<UNK>，也就是都把这些生僻字当作同一个<UNK>
        # 但是embedding里是保留有每个生僻字的向量的，另外，embedding的大小(长度，而非维度)比字符数大一点
        # 猜测：虽然在lookup的时候可以保证对应的字取到对应的embedding，但是反向传播的时候，可能会波及毫不相关的embedding的
        # 依据：当我用一个比较大的字库时，同样的语料，收敛速度变慢了(原本的字库是只包含语料内的文字)，
        #       表面上是参数变多了(因为多了一些理论上不相关的embedding)，
        #       但如果训练的时候，反向传播时如果没有涉及这些不相关的embedding，那应该是没影响的。
        for i in range(len(v_x[0])):
            for j, n in enumerate(v_x[0][i]):
                if n in unk_chars:
                    v_x[0][i][j] = 1

        for i in range(len(t_x[0])):
            for k in range(len(t_x[0][i])):
                for j, n in enumerate(t_x[0][i][k]):
                    if n in unk_chars:
                        t_x[0][i][k][j] = 1

        transducer_dict = None
        if transducer is not None:
            char2idx = {k:v for v, k in list(idx2char.items())}

            def transducer_dict(trans_str):
                return self.define_transducer_dict(trans_str, char2idx, sess[-1], transducer)
        print('model.train内，进入训练周期前耗时：%d' % int(time() - t_pre))
        print('开始进入epoch周期')
        for epoch in range(epochs):
            print('epoch: %d' % (epoch + 1))
            t = time()
            if epoch % decay_step == 0 and decay > 0:
                lr_r = lr/(1 + decay*(epoch/decay_step))

            data_list = t_x + t_y

            samples = list(zip(*data_list))

            random.shuffle(samples)
            # 这里的samples就是buckets
            for sample in samples:
                c_len = len(sample[0][0])
                idx = self.bucket_dit[c_len]
                real_batch_size = self.real_batches[idx]
                model = self.input_v[idx] + self.output_[idx]
                Batch.train(sess=sess[0], model=model, batch_size=real_batch_size, config=self.train_step[idx],
                            lr=self.l_rate, lrv=lr_r, dr=self.drop_out, drv=self.drop_out_v, data=list(sample), verbose=False)

            predictions = []

            #for v_b_x in zip(*v_x):
            c_len = len(v_x[0][0])
            idx = self.bucket_dit[c_len]
            b_prediction = self.predict(data=v_x, sess=sess, model=self.input_v[idx] + self.output[idx], index=idx, argmax=True, batch_size=200)
            b_prediction = toolbox.decode_tags(b_prediction, idx2tag)
            predictions.append(b_prediction)

            predictions = list(zip(*predictions))
            predictions = toolbox.merge_bucket(predictions)

            if self.is_space == 'sea':
                prediction_out, raw_out = toolbox.generate_output_sea(chars, predictions)
            else:
                prediction_out, raw_out = toolbox.generate_output(chars, predictions, trans_dict, transducer_dict)

            if sent_seg:
                scores = evaluation.evaluator(prediction_out, v_y_gold, raw_out, v_y_raw)
            else:
                scores = evaluation.evaluator(prediction_out, v_y_gold)
            if sent_seg:
                c_score = scores[2] * scores[5]
                c_best_score = best_score[2] * best_score[5]
            else:
                c_score = scores[2]
                c_best_score = best_score[2]

            if c_score > c_best_score:
                best_epoch = epoch + 1
                best_score = scores
                self.saver.save(sess[0], trained_model, write_meta_graph=False)

                #  保存最好分数，方便下次重启时比较
                print('保存最好分数')
                score_file = open(trained_model + '_score', 'wb')
                pickle.dump(best_score, score_file)
                score_file.close()

                if outpath is not None:
                    wt = codecs.open(outpath, 'w', encoding='utf-8')
                    for pre in prediction_out[0]:
                        wt.write(pre + '\n')
                    wt.close()


            if sent_seg:
                print('Sentence segmentation:')
                print('F score: %f\n' % scores[5])
                print('Word segmentation:')
                print('F score: %f' % scores[2])
            else:
                print('F score: %f | P score: %f | R score: %f' % (c_score, scores[0], scores[1]))
            print('Time consumed: %d seconds' % int(time() - t))
        print('Training is finished!')
        if sent_seg:
            print('Sentence segmentation:')
            print('Best F score: %f' % best_score[5])
            print('Best Precision: %f' % best_score[3])
            print('Best Recall: %f\n' % best_score[4])
            print('Word segmentation:')
            print('Best F score: %f' % best_score[2])
            print('Best Precision: %f' % best_score[0])
            print('Best Recall: %f\n' % best_score[1])
        else:
            print('Best F score: %f' % best_score[2])
            print('Best Precision: %f' % best_score[0])
            print('Best Recall: %f\n' % best_score[1])
        print('Best epoch: %d' % best_epoch)

    def test(self, t_x, t_y_raw, t_y_gold, idx2tag, idx2char, unk_chars, trans_dict, sess, transducer, ensemble=None,
             batch_size=100, bias=-1, outpath=None):

        chars = toolbox.decode_chars(t_x[0], idx2char)
        gold_out = t_y_gold

        for i in range(len(t_x[0])):
            for j, n in enumerate(t_x[0][i]):
                if n in unk_chars:
                    t_x[0][i][j] = 1

        transducer_dict = None
        if transducer is not None:
            char2idx = {v: k for k, v in list(idx2char.items())}

            def transducer_dict(trans_str):
                return self.define_transducer_dict(trans_str, char2idx, sess[-1], transducer)

        if bias < 0:
            argmax = True
        else:
            argmax = False

        prediction = self.predict(data=t_x, sess=sess, model=self.input_v[0] + self.output[0], index=0,
                                  argmax=argmax, batch_size=batch_size, ensemble=ensemble)

        if bias >= 0 and self.crf == 0:
            prediction = [toolbox.biased_out(prediction[0], bias)]

        predictions = toolbox.decode_tags(prediction, idx2tag)

        if self.is_space == 'sea':
            prediction_out, raw_out = toolbox.generate_output_sea(chars, predictions)
        else:
            prediction_out, raw_out = toolbox.generate_output(chars, predictions, trans_dict, transducer_dict)

        scores = evaluation.evaluator(prediction_out, gold_out, verbose=True)

        if outpath is not None:
            wt = codecs.open(outpath, 'w', encoding='utf-8')
            for pre in prediction_out[0]:
                wt.write(pre + '\n')
            wt.close()

        print('Evaluation scores:')
        print('Precision: %f' % scores[0])
        print('Recall: %f' % scores[1])
        print('F score: %f' % scores[2])
        print('True negative rate: %f' % scores[3])

    def tag(self, r_x, r_x_raw, idx2tag, idx2char, unk_chars, trans_dict, sess, transducer, ensemble=None, batch_size=100,
            outpath=None, seg_large=False):

        chars = toolbox.decode_chars(r_x[0], idx2char)

        for i in range(len(r_x[0])):
            for j, n in enumerate(r_x[0][i]):
                if n in unk_chars:
                    r_x[0][i][j] = 1

        c_len = len(r_x[0][0])
        idx = self.bucket_dit[c_len]

        real_batch = int(batch_size * 300 / c_len)

        transducer_dict = None
        if transducer is not None:
            char2idx = {v: k for k, v in list(idx2char.items())}

            def transducer_dict(trans_str):
                return self.define_transducer_dict(trans_str, char2idx, sess[-1], transducer)

        prediction = self.predict(data=r_x, sess=sess, model=self.input_v[idx] + self.output[idx], index=idx,
                                    argmax=True, batch_size=real_batch, ensemble=ensemble)

        predictions = toolbox.decode_tags(prediction, idx2tag)

        if self.is_space == 'sea':
            prediction_out, raw_out = toolbox.generate_output_sea(chars, predictions)
            multi_out = prediction_out
        else:
            prediction_out, raw_out, multi_out = toolbox.generate_output(chars, predictions, trans_dict, transducer_dict, multi_tok=True)

        pre_out = []
        mut_out = []
        for pre in prediction_out:
            pre_out += pre
        for mul in multi_out:
            mut_out += mul
        prediction_out = pre_out
        multi_out = mut_out

        prediction_out = toolbox.mlp_post(r_x_raw, prediction_out)

        if not seg_large and outpath is not None:
            toolbox.printer(r_x_raw, prediction_out, multi_out, outpath)
        else:
            return prediction_out, multi_out

    def predict(self, data, sess, model, index=None, argmax=True, batch_size=100, ensemble=None, verbose=False):
        if self.crf:
            assert index is not None
            predictions = Batch.predict(sess=sess[0], decode_sess=sess[1], model=model, transitions=[self.transition_char], crf=self.crf,
                                        scores=self.scores[index], decode_holders=self.decode_holders[index], batch_size=batch_size,
                                        data=data, dr=self.drop_out, ensemble=ensemble, verbose=verbose)
        else:
            predictions = Batch.predict(sess=sess[0], model=model, crf=self.crf, argmax=argmax, batch_size=batch_size, data=data,
                                        dr=self.drop_out, ensemble=ensemble, verbose=verbose)
        return predictions

