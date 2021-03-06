import codecs
import random


# 提取字典，并且统计字频，在当前目录生成chars.txt，filelist是文件path数组
# 排除exclude_file里收录的字符，并且排除频数低于unk_rule的字符
def get_chars(filelist, out_path='./chars.txt', exclude_file=None, unk_rule=1):
    # 读取excludeFile，加入到excludeSet
    # excludeFile一行一个字符
    exclude_set = None
    if exclude_file is not None:
        exclude_set = set()
        for line in codecs.open(exclude_file, 'rb', encoding='utf-8'):
            char = line.strip().replace(' ', '')
            exclude_set.add(char)

    char_set = {' ': 1}  # (普通)空格作为字典的第一个字，并且计数设定为1，把它当作生僻字来处理
    out_char = codecs.open(out_path, 'w', encoding='utf-8')
    for file_name in filelist:
        for line in codecs.open(file_name, 'rb', encoding='utf-8'):
            line = line.strip().replace(' ', '')
            for ch in line:
                # 如果在excludeSet，就不加入到字典
                if exclude_set is not None and ch in exclude_set:
                    continue
                if ch in char_set:
                    char_set[ch] += 1
                else:
                    char_set[ch] = 1
    for k, v in list(char_set.items()):
        if v < unk_rule:
            continue
        out_char.write(k + '\t' + str(v) + '\n')
    out_char.close()


def divide_corpus(all_path, train_path, dev_path, test_path, train_rate=0.6, dev_rate=0.2, test_rate=0.2):
    all = codecs.open(all_path, 'r', encoding='utf-8')
    train = codecs.open(train_path, 'w', encoding='utf-8')
    dev = codecs.open(dev_path, 'w', encoding='utf-8')
    test = codecs.open(test_path, 'w', encoding='utf-8')
    # 区间顺序是train，dev，test
    random.seed()
    for line in all:
        ra = random.random()
        if ra <= train_rate:
            train.write(line)
        elif ra <= train_rate + dev_rate:
            dev.write(line)
        else:
            test.write(line)

    all.close()
    train.close()
    dev.close()
    test.close()


if __name__ == '__main__':
    print('测试')
    divide_corpus('./data/people2014.txt', './data/people2014/people2014_train.segd',
                  './data/people2014/people2014_dev.segd', './data/people2014/people2014_test.segd')