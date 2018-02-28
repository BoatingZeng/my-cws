import codecs
from tf_cws.tagger import Tagger


if __name__ == '__main__':
    print('测试')
    tagger = Tagger('./data/pku', sent_limit=20)
    lines = codecs.open('./data/test_raw.txt', 'rb', encoding='utf-8')
    seg_out, token_out = tagger.tag(lines, isTokenize=True)
    print(seg_out)
    print(token_out)
