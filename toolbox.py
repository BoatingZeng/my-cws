import codecs
import os
import numpy as np
import math


def get_gold(sent):
    line = ''
    for tk in sent:
        line += '  ' + tk[1]
    return line[2:]


def get_tags(can, tag_scheme='BIES'):
    tags = []
    if tag_scheme == 'BI':
        for i in range(len(can)):
            if i == 0:
                tags.append('B')
            else:
                tags.append('I')
    else:
        for i in range(len(can)):
            if len(can) == 1:
                tags.append('S')
            elif i == 0:
                tags.append('B')
            elif i == len(can) - 1:
                tags.append('E')
            else:
                tags.append('I')
    return tags


def raw2tags(raw, sents, path, train_file, gold_path=None, reset=False, tag_scheme='BIES'):
    wt = codecs.open(path + '/' + train_file, 'w', encoding='utf-8')
    wg = None
    if gold_path is not None:
        wg = codecs.open(path + '/' + gold_path, 'w', encoding='utf-8')
    wtg = None
    if reset or not os.path.isfile(path + '/tags.txt'):
        wtg = codecs.open(path + '/tags.txt', 'w', encoding='utf-8')
    assert len(raw) == len(sents)
    invalid = 0
    s_tags = set()

    def matched(can, sent_l, tags):
        tags += get_tags(can, tag_scheme=tag_scheme)
        sent_l.pop(0)
        return tags

    for raw_l, sent_l in zip(raw, sents):
        tags = []
        cans = raw_l.split(' ')
        gold = get_gold(sent_l)
        pre = ''
        for can in cans:
            t_can = can.strip()
            purged = len(can) - len(t_can)
            if purged > 0:
                can = t_can
            while purged > 0:
                if tag_scheme == 'BI':
                    tags.append('I')
                else:
                    tags.append('X')
                purged -= 1
            done = False
            if len(pre) > 0:
                can = pre + ' ' + can
            while not done:
                if can == sent_l[0][1]:
                    tags = matched(can, sent_l, tags)
                    done = True
                    pre = ''
                else:
                    if len(can) >= len(sent_l[0][1]):
                        s_l = len(sent_l[0][1])
                        s_can = can[:s_l]
                        if s_can != sent_l[0][1]:
                            done = True
                        tags = matched(s_can, sent_l, tags)
                        can = can[s_l:]
                        if len(can) == 0:
                            done = True
                            pre = ''
                    else:
                        pre = can
                        done = True
            if len(pre) == 0:
                if tag_scheme == 'BI':
                    tags.append('I')
                else:
                    tags.append('X')
        if len(tags) > 0:
            tags.pop()
        if len(tags) == len(raw_l):
            for tg in tags:
                s_tags.add(tg)
            wt.write(raw_l + '\n')
            wt.write(''.join(tags) + '\n')
            wt.write('\n')
        else:
            invalid += 1
        if wg is not None:
            wg.write(gold + '\n')
    if wg is not None:
        wg.close()
    if wtg is not None:
        for stg in s_tags:
            wtg.write(stg + '\n')
        wtg.close()
    wt.close()
    print('invalid sentences: ', invalid, len(raw))


def get_chars(path, filelist):
    char_set = {}
    out_char = codecs.open(path + '/chars.txt', 'w', encoding='utf-8')
    for i, file_name in enumerate(filelist):
        for line in codecs.open(path + '/' + file_name, 'rb', encoding='utf-8'):
            line = line.strip()
            for ch in line:
                if ch in char_set:
                    if i == 0:
                        char_set[ch] += 1
                else:
                    char_set[ch] = 1
    for k, v in list(char_set.items()):
        out_char.write(k + '\t' + str(v) + '\n')
    out_char.close()


def get_dicts(path, tag_scheme='BIES', crf=1):
    char2idx = {'<P>': 0, '<UNK>': 1, '<#>': 2}
    '''
        <P>：每个bucket里填充的空白
        <UNK>：生僻字，或者字典里没有的字
        <#>：句子被截断时，后半句的开始
    '''
    unk_chars = []
    idx = 3
    for line in codecs.open(path + '/chars.txt', 'r', encoding='utf-8'):
        segs = line.split('\t')
        if len(segs[0].strip()) == 0:
            if ' ' not in char2idx:
                char2idx[' '] = idx
                idx += 1
        else:
            char2idx[segs[0]] = idx
            if int(segs[1]) == 1:
                unk_chars.append(idx)
            idx += 1
    idx2char = {k: v for v, k in list(char2idx.items())}
    if tag_scheme == 'BI':
        if crf > 0:
            tag2idx = {'<P>': 0, 'B': 1, 'I': 2}
        else:
            tag2idx = {'B': 0, 'I': 1}
    else:
        if crf > 0:
            tag2idx = {'<P>': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
            idx = 5
        else:
            tag2idx = {'B': 0, 'I':1, 'E':2, 'S':3}
            idx = 4
        for line in codecs.open(path + '/tags.txt', 'r', encoding='utf-8'):
            line = line.strip()
            if line not in tag2idx:
                tag2idx[line] = idx
                idx += 1
    idx2tag = {k: v for v, k in list(tag2idx.items())}

    return char2idx, unk_chars, idx2char, tag2idx, idx2tag


def get_sample_embedding(path, emb, chars2idx):
    chars = list(chars2idx.keys())
    short_emb = emb[emb.index('/') + 1: emb.index('.')]
    emb_dic = {}
    valid_chars=[]
    for line in codecs.open(emb, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        emb_dic[sets[0]] = np.asarray(sets[1:], dtype='float32')
    fout = codecs.open(path + '/' + short_emb + '_sub.txt', 'w', encoding='utf-8')
    for ch in chars:
        p_line = ch
        if ch in emb_dic:
            valid_chars.append(ch)
            for emb in emb_dic[ch]:
                p_line += ' ' + str(emb)
            fout.write(p_line + '\n')
    fout.close()


def read_sample_embedding(path, short_emb, char2idx):
    emb_values = []
    valid_chars = []
    emb_dic={}
    for line in codecs.open(path + '/' + short_emb + '_sub.txt', 'rb', encoding='utf-8'):
        first_ch = line[0]
        line = line.rstrip()
        sets = line.split(' ')
        if first_ch == ' ':
            emb_dic[' '] = np.asarray(sets, dtype='float32')
        else:
            emb_dic[sets[0]] = np.asarray(sets[1:], dtype='float32')
    emb_dim = len(list(emb_dic.items())[0][1])
    for ch in list(char2idx.keys()):
        if ch in emb_dic:
            emb_values.append(emb_dic[ch])
            valid_chars.append(ch)
        else:
            rand = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
            emb_values.append(np.asarray(rand, dtype='float32'))
    emb_dim = len(emb_values[0])
    return emb_dim, emb_values, valid_chars


def get_input_vec(path, fname, char2idx, tag2idx, limit=500, train_size=-1):
    ct = 0
    max_len = 0

    x_indices = []
    y_indices = []
    s_count = 0
    l_count = 0
    x = []
    y = []

    n_sent = 0

    for line in codecs.open(path + '/' + fname, 'r', encoding='utf-8'):
        line = line.strip()
        if len(line) == 0:
            ct = 0
        elif ct == 0:
            max_len = max(max_len, len(line))
            s_count += 1
            if len(line) > limit:
                l_count += 1
            chopped = False
            while len(line) > 0:
                s_line = line[:limit - 1]
                line = line[limit - 10:]
                if len(line) < 10:
                    line = ''
                if not chopped:
                    chopped = True
                else:
                    x.append(char2idx['<#>'])
                for ch in s_line:
                    if len(ch.strip()) == 0:
                        x.append(char2idx[' '])
                    elif ch in char2idx:
                        x.append(char2idx[ch])
                    else:
                        x.append(char2idx['<UNK>'])
                x_indices.append(x)
                x = []
            ct = 1
        elif ct == 1:
            chopped = False
            while len(line) > 0:
                s_line = line[:limit - 1]
                line = line[limit - 10:]
                if len(line) < 10:
                    line = ''
                if not chopped:
                    chopped = True
                else:
                    y.append(tag2idx['I'])
                for ch in s_line:
                    y.append(tag2idx[ch])
                y_indices.append(y)
                y = []
            n_sent += 1
        if 0 < train_size <= n_sent:
            break
    max_len = min(max_len, limit)
    if l_count > 0:
        print('%d (out of %d) sentences are chopped.' % (l_count, s_count))
    return [x_indices], [y_indices], max_len


def chop(line, ad_s, limit):
    out = []
    chopped = False
    while len(line) > 0:
        if chopped:
            s_line = line[:limit - 1]
            s_line = [ad_s] + s_line
        else:
            chopped = True
            s_line = line[:limit]
        out.append(s_line)
        line = line[limit - 10:]
        if len(line) < 10:
            line = ''
    while len(out) > 0 and len(out[-1]) < limit-1:
        out[-1].append(0)
    return out


def get_input_vec_raw(path, fname, char2idx, lines=None, limit=500):
    max_len = 0
    x_indices = []
    s_count = 0
    l_count = 0
    x = []
    if lines is None:
        assert fname is not None
        if path is None:
            real_path = fname
        else:
            real_path = path + '/' + fname
        lines = codecs.open(real_path, 'r', encoding='utf-8')

    for line in lines:
        line = line.strip()
        if len(line) > 0:
            max_len = max(max_len, len(line))
            s_count += 1

            for ch in line:
                if len(ch.strip()) == 0:
                    x.append(char2idx[' '])
                elif ch in char2idx:
                    x.append(char2idx[ch])
                else:
                    x.append(char2idx['<UNK>'])

            if len(line) > limit:
                l_count += 1
                chop_x = chop(x, char2idx['<#>'], limit)
                x_indices += chop_x
            else:
                x_indices.append(x)
            x = []
    max_len = min(max_len, limit)
    if l_count > 0:
        pass
        # print('%d (out of %d) sentences are chopped.' % (l_count, s_count))
    return [x_indices], max_len


def buckets(x, y, size=50):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    samples = x + y
    num_items = len(samples)
    xy = list(zip(*samples))
    xy.sort(key=lambda i: len(i[0]))
    t_len = size
    idx = 0
    bucks = [[[]] for _ in range(num_items)]
    for item in xy:
        if len(item[0]) > t_len:
            if len(bucks[0][idx]) > 0:
                for buck in bucks:
                    buck.append([])
                idx += 1
            while len(item[0]) > t_len:
                t_len += size
        for i in range(num_items):
            #print item[i]
            bucks[i][idx].append(item[i])

    return bucks[:num_inputs], bucks[num_inputs:]


def pad_bucket(x, y, limit, bucket_len_c=None):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    num_tags = len(y)
    padded = [[] for _ in range(num_tags + num_inputs)]
    bucket_counts = []
    samples = x + y
    xy = list(zip(*samples))
    if bucket_len_c is None:
        bucket_len_c = []
        for i, item in enumerate(xy):
            print(len(item))
            print(len(item[0]))
            max_len = len(item[0][-1])
            if i == len(xy) - 1:
                max_len = limit
            bucket_len_c.append(max_len)
            bucket_counts.append(len(item[0]))
            for idx in range(num_tags + num_inputs):
                padded[idx].append(pad_zeros(item[idx], max_len))
        print('Number of buckets: ', len(bucket_len_c))
    else:
        idy = 0
        for item in xy:
            max_len = len(item[0][-1])
            while idy < len(bucket_len_c) and max_len > bucket_len_c[idy]:
                idy += 1
            bucket_counts.append(len(item[0]))
            if idy >= len(bucket_len_c):
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], max_len))
                bucket_len_c.append(max_len)
            else:
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], bucket_len_c[idy]))
    return padded[:num_inputs], padded[num_inputs:], bucket_len_c, bucket_counts


def pad_zeros(l, max_len):
    padded = None
    if type(l) is list:
        padded = []
        for item in l:
            if len(item) <= max_len:
                padded.append(np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0))
            else:
                padded.append(np.asarray(item[:max_len]))
        padded = np.asarray(padded)
    elif type(l) is dict:
        padded = {}
        for k, v in l.items():
            padded[k] = [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0) for item in v]
    return padded


def get_real_batch(counts, b_size):
    real_batch_sizes = []
    for c in counts:
        if c < b_size:
            real_batch_sizes.append(c)
        else:
            real_batch_sizes.append(b_size)
    return real_batch_sizes


def get_new_embeddings(new_chars, emb_dim, emb_path):
    assert os.path.isfile(emb_path)
    emb = {}
    new_emb = []
    for line in codecs.open(emb_path, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        emb[sets[0]] = np.asarray(sets[1:], dtype='float32')
    if '<UNK>' not in emb:
        unk = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        emb['<UNK>'] = np.asarray(unk, dtype='float32')
    for ch in new_chars:
        if ch in emb:
            new_emb.append(emb[ch])
        else:
            new_emb.append(emb['<UNK>'])
    return new_emb


def decode_chars(idx, idx2chars):
    out = []
    for line in idx:
        line = np.trim_zeros(line)
        out.append([idx2chars[item] for item in line])
    return out


def decode_tags(idx, index2tags):
    out = []
    for id in idx:
        sents = []
        for line in id:
            sent = []
            for item in line:
                tag = index2tags[item]
                tag = tag.replace('E', 'I')
                tag = tag.replace('S', 'B')
                tag = tag.replace('J', 'Z')
                tag = tag.replace('D', 'K')
                sent.append(tag)
            sents.append(sent)
        out.append(sents)
    return out


def merge_bucket(x):
    out = []
    for item in x:
        m = []
        for i in item:
            m += i
        out.append(m)
    return out


def biased_out(prediction, bias):
    out = []
    b_pres = []
    for pre in prediction:
        b_pres.append(pre[:,0] - pre[:,1])
    props = np.concatenate(b_pres)
    props = np.sort(props)[::-1]
    idx = int(bias*len(props))
    if idx == len(props):
        idx -= 1
    th = props[idx]
    print('threshold: ', th, 1 / (1 + np.exp(-th)))
    for pre in b_pres:
        pre[pre >= th] = 0
        pre[pre != 0] = 1
        out.append(pre)
    return out


def generate_output_sea(chars, tags):
    out = []
    raw_out = []
    sent_seg = False

    def split_sent(lines, s_str):
        for i in range(len(lines)):
            s_line = lines[i].strip()
            while s_line and s_line[-1] == s_str:
                s_line = s_line[:-1]
            sents = s_line.split(s_str)
            lines[i] = [sent.strip() for sent in sents]
        return lines

    for i, tag in enumerate(tags):
        assert len(chars) == len(tag)
        sub_out = []
        sub_raw_out = []
        j_chars = []
        j_tags = []
        is_first = True
        for chs, tgs in zip(chars, tag):
            if chs[0] == '<#>':
                assert len(j_chars) > 0
                if is_first:
                    is_first = False
                    j_chars[-1] = j_chars[-1][:-5] + chs[6:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[6:]
                else:
                    j_chars[-1] = j_chars[-1][:-5] + chs[5:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[5:]
            else:
                j_chars.append(chs)
                j_tags.append(tgs)
                is_first = True
        chars = j_chars
        tag = j_tags
        for chs, tgs in zip(chars, tag):
            assert len(chs) == len(tgs)
            p_line = ''
            r_line = ''
            for ch, tg in zip(chs, tgs):
                r_line += ' ' + ch
                if tg == 'I':
                    if ch == '.' or (ch >= '0' and ch <= '9'):
                        p_line += ch
                    else:
                        p_line += ' ' + ch
                elif tg == 'B':
                    p_line += '  ' + ch
                elif tg == 'T':
                    sent_seg = True
                    p_line += '  ' + ch + '<SENT>'
                    r_line += '<SENT>'
                elif tg == 'U':
                    sent_seg = True
                    p_line += ch + '<SENT>'
                    r_line += '<SENT>'
                elif len(ch.strip()) > 0:
                    p_line += '  ' + ch
            sub_out.append(p_line.strip())
            sub_raw_out.append(r_line.strip())
        out.append(sub_out)
        raw_out.append(sub_raw_out)
    out[0][-1].rstrip('<SENT>')
    raw_out[0][-1].rstrip('<SENT>')
    if sent_seg:
        out = split_sent(out[0], '<SENT>')
        raw_out = split_sent(raw_out[0], '<SENT>')
    return out, raw_out


def generate_output(chars, tags, trans_dict, transducer_dict=None, multi_tok=False):
    out = []
    mult_out = []
    raw_out = []
    sent_seg = False

    def map_trans(c_trans):
        if c_trans in trans_dict:
            c_trans = trans_dict[c_trans]
        elif c_trans.lower() in trans_dict:
            c_trans = trans_dict[c_trans.lower()]
        elif transducer_dict is not None:
            c_trans = transducer_dict(c_trans)
        c_trans = c_trans.replace('    ', '  ')
        c_trans = c_trans.replace('   ', '  ')
        return c_trans

    def add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=False):
        c_trans = c_trans.strip()
        if len(c_trans) > 0:
            if trans:
                o_trans = c_trans
                c_trans = map_trans(c_trans)
                if multi_tok:
                    num_tr = len(c_trans.split('  '))
                    mt_p_line += '  ' + o_trans + '!#!' + str(num_tr) + '  ' + c_trans
            else:
                if multi_tok:
                    mt_p_line += '  ' + c_trans
            p_line += '  ' + c_trans
        return p_line, mt_p_line

    def split_sent(lines, s_str):
        for i in range(len(lines)):
            s_line = lines[i].strip()
            while s_line and s_line[-1] == s_str:
                s_line = s_line[:-1]
            sents = s_line.split(s_str)
            lines[i] = [sent.strip() for sent in sents]
        return lines

    for i, tag in enumerate(tags):
        assert len(chars) == len(tag)
        sub_out = []
        sub_raw_out = []
        multi_sub_out = []
        j_chars = []
        j_tags = []
        is_first = True
        for chs, tgs in zip(chars, tag):
            if chs[0] == '<#>':
                assert len(j_chars) > 0
                if is_first:
                    is_first = False
                    j_chars[-1] = j_chars[-1][:-5] + chs[6:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[6:]
                else:
                    j_chars[-1] = j_chars[-1][:-5] + chs[5:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[5:]
            else:
                j_chars.append(chs)
                j_tags.append(tgs)
                is_first = True
        chars = j_chars
        tag = j_tags
        for chs, tgs in zip(chars, tag):
            assert len(chs) == len(tgs)
            c_word = ''
            c_trans = ''
            p_line = ''
            r_line = ''
            mt_p_line = ''
            for ch, tg in zip(chs, tgs):
                r_line += ch
                if tg == 'I':
                    if len(c_trans) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                        c_word = ch
                    else:
                        c_word += ch
                elif tg == 'Z':
                    if len(c_word) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                        c_trans = ch
                    else:
                        c_trans += ch
                elif tg == 'B':
                    if len(c_word) > 0:
                        c_word = c_word.strip()
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                    elif len(c_trans) > 0:
                        c_trans = c_trans.strip()
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                    c_word = ch
                elif tg == 'K':
                    if len(c_word) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                    elif len(c_trans) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                    c_trans = ch
                elif tg == 'T':
                    sent_seg = True
                    if len(c_word) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                    elif len(c_trans) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                    p_line += '  ' + ch + '<SENT>'
                    if multi_tok:
                        mt_p_line += '  ' + ch + '<SENT>'
                    r_line += '<SENT>'
                elif tg == 'U':
                    sent_seg = True
                    if len(c_word) > 0:
                        c_word += ch
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                    elif len(c_trans) > 0:
                        c_trans += ch
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                    elif len(ch.strip()) > 0:
                        p_line += ch
                        if multi_tok:
                            mt_p_line += ch
                    p_line += '<SENT>'
                    if multi_tok:
                        mt_p_line += '<SENT>'
                    r_line += '<SENT>'
                elif tg == 'X' and len(ch.strip()) > 0:
                    if len(c_word) > 0:
                        c_word += ch
                    elif len(c_trans) > 0:
                        c_trans += ch
                    else:
                        c_word = ch
                elif len(ch.strip()) > 0:
                    if len(c_word) > 0:
                        c_word += '  ' + ch
                    elif len(c_trans) > 0:
                        c_trans += '  ' + ch
                    else:
                        c_word = ch
            if len(c_word) > 0:
                c_word = c_word.strip()
                p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
            elif len(c_trans) > 0:
                c_trans = c_trans.strip()
                p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
            sub_out.append(p_line.strip())
            sub_raw_out.append(r_line.strip())
            if multi_tok:
                multi_sub_out.append(mt_p_line.strip())
        out.append(sub_out)
        raw_out.append(sub_raw_out)
        if multi_tok:
            mult_out.append(multi_sub_out)
    out[0][-1].rstrip('<SENT>')
    raw_out[0][-1].rstrip('<SENT>')
    if sent_seg:
        out = split_sent(out[0], '<SENT>')
        raw_out = split_sent(raw_out[0], '<SENT>')
    if multi_tok:
        mult_out[0][-1].rstrip('<SENT>')
        if sent_seg:
            mult_out = split_sent(mult_out[0], '<SENT>')
        return out, raw_out, mult_out
    else:
        return out, raw_out


def mlp_post(raw, prediction):
    assert len(raw) == len(prediction)
    out = []
    for r_l, p_l in zip(raw, prediction):
        st = ''
        rtokens = r_l.split()
        ptokens = p_l.split('  ')
        purged = []
        for pt in ptokens:
            purged.append(pt.strip())
        ptokens = purged
        ptokens_str = ''.join(ptokens)
        assert ''.join(rtokens) == ''.join(ptokens_str.split())
        for p_t in ptokens:
            st += p_t + ' '
        out.append(st.strip())
    return out


def validator(raw, generated):
    raw_l = ''.join(raw)
    raw_l = ''.join(raw_l.split())
    for g in generated:
        g_tokens = g.split('  ')
        j = 0
        while j < len(g_tokens):
            if '!#!' in g_tokens[j]:
                segs = g_tokens[j].split('!#!')
                c_t = int(segs[1])
                r_seg = ''.join(segs[0].split())
                l_w = len(r_seg)
                if r_seg == raw_l[:l_w]:
                    raw_l = raw_l[l_w:]
                    raw_l = raw_l.strip()
                else:
                    raise Exception('Error: unmatch...')
                j += c_t
            else:
                r_seg = ''.join(g_tokens[j].split())
                l_w = len(r_seg)
                if r_seg == raw_l[:l_w]:
                    raw_l = raw_l[l_w:]
                    raw_l = raw_l.strip()
                else:
                    print(r_seg)
                    print(raw_l[:l_w])
                    print('')
                    raise Exception('Error: unmatch...')
            j += 1



def printer(raw, tagged, multi_out, outpath):
    assert len(tagged) == len(multi_out)
    validator(raw, multi_out)
    wt = codecs.open(outpath, 'w', encoding='utf-8')
    for tg in tagged:
        wt.write(tg + '\n')
    wt.close()


def viterbi(max_scores, max_scores_pre, length, batch_size):
    best_paths = []
    for m in range(batch_size):
        path = []
        last_max_node = np.argmax(max_scores[m][length[m] - 1])
        path.append(last_max_node)
        for t in range(1, length[m])[::-1]:
            last_max_node = max_scores_pre[m][t][last_max_node]
            path.append(last_max_node)
        path = path[::-1]
        best_paths.append(path)
    return best_paths


def trim_output(out, length):
    assert len(out) == len(length)
    trimmed_out = []
    for item, l in zip(out, length):
        trimmed_out.append(item[:l])
    return trimmed_out


def read_ngrams(path, ng):
    ngs = []
    for i in range(2, ng + 1):
        ng = {}
        for line in codecs.open(path + '/' + str(i) + 'gram.txt', 'r', encoding='utf-8'):
            line = line.rstrip()
            segs = line.split('\t')
            while len(segs[0]) < i:
                segs[0] += ' '
            ng[segs[0]] = int(segs[1])
        ngs.append(ng)
    return ngs


def get_new_chars(path, char2idx):
    new_chars = set()
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        for ch in line:
            if ch not in char2idx:
                new_chars.add(ch)
    return new_chars


def get_valid_chars(chars, emb_path):
    valid_chars = []
    total = []
    for line in codecs.open(emb_path, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        total.append(sets[0])
    for ch in chars:
        if ch in total:
            valid_chars.append(ch)
    return valid_chars


def update_char_dict(char2idx, new_chars, unk_chars, valid_chars=None):
    dim = len(char2idx) + 10
    if valid_chars is not None:
        for ch in valid_chars:
            if ch in unk_chars:
                unk_chars.remove(ch)
    for char in new_chars:
        if char not in char2idx and len(char.strip()) > 0:
            char2idx[char] = dim
            if valid_chars is None or char not in valid_chars:
                unk_chars.append(dim)
            dim += 1
    idx2char = {k: v for v, k in list(char2idx.items())}
    return char2idx, idx2char, unk_chars


def get_input_vec_tag(path, fname, char2idx, lines=None, limit=500):
    x_indices = []
    out = []
    x = []
    if lines is None:
        assert fname is not None
        if path is None:
            real_path = fname
        else:
            real_path = path + '/' + fname
        lines = codecs.open(real_path, 'r', encoding='utf-8')
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if len(line) > 0:
                for ch in line:
                    if len(ch.strip()) == 0:
                        x.append(char2idx[' '])
                    elif ch in char2idx:
                        x.append(char2idx[ch])
                    else:
                        x.append(char2idx['<UNK>'])
                x_indices += x
                x = []
            elif len(x_indices) > 0:
                x_indices = chop(x_indices, char2idx['<#>'], limit)
                out += x_indices
                x_indices = []
                is_first = True

    if len(x_indices) > 0:
        x_indices = chop(x_indices, char2idx['<#>'], limit)
        out += x_indices

    return [out], limit