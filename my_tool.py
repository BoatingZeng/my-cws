import codecs

# 提取字典，并且统计字频，在当前目录生成chars.txt，filelist是文件path数组
# TODO 字典里不保留英文字母，特殊字符，这个要定个列表来做过滤
def get_chars(filelist):
    char_set = {' ': 1}  # (普通)空格作为字典的第一个字，并且计数设定为1，把它当作生僻字来处理
    out_char = codecs.open('./chars.txt', 'w', encoding='utf-8')
    for i, file_name in enumerate(filelist):
        for line in codecs.open(file_name, 'rb', encoding='utf-8'):
            line = line.strip().replace(' ', '')
            for ch in line:
                if ch in char_set:
                    if i == 0:
                        char_set[ch] += 1
                else:
                    char_set[ch] = 1
    for k, v in list(char_set.items()):
        out_char.write(k + '\t' + str(v) + '\n')
    out_char.close()