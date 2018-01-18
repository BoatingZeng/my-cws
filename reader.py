import codecs


def get_raw(path, fin, fout, is_dev=True):
    fout = codecs.open(path + '/' + fout, 'w', encoding='utf-8')
    fout_dev = None
    if not is_dev:
        fout_dev = codecs.open(path + '/raw_dev.txt', 'w', encoding='utf-8')
    cter = 0
    for line in codecs.open(path + '/' + fin, 'r', encoding='utf-8'):
        line = line.strip()
        if len(line) > 0:
            line = ''.join(line.split())
            if not is_dev and cter == 9:
                fout_dev.write(line + '\n')
                cter = 0
            else:
                fout.write(line + '\n')
                cter += 1

    fout.close()
    if not is_dev:
        fout_dev.close()


def raw(path):
    sents = []
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        sents.append(line)
    return sents


def gold(path, is_dev=True):
    sents = []
    sent = []
    cter = 0
    sents_dev = None
    if not is_dev:
        sents_dev = []
    for line in codecs.open(path, 'rb', encoding='utf8'):
        line = line.strip()
        if len(line) > 0:
            segs = line.split()
            for i, seg in enumerate(segs):
                sent.append((str(i + 1), seg))
            if not is_dev and cter == 9:
                sents_dev.append(sent)
                cter = 0
            else:
                sents.append(sent)
                cter += 1
            sent = []
    if is_dev:
        return sents
    else:
        return sents, sents_dev


def test_gold(path):
    sents = []
    st = ''
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        if len(line) > 0:
            segs = line.split()
            for seg in segs:
                st += '  ' + seg
            sents.append(st.strip())
            st = ''
    return sents