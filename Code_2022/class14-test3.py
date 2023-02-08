#!/usr/bin/env python
# encoding: utf-8
'''
@author: MrYx
@github: https://github.com/MrYxJ
'''

import jieba
from gensim.models import word2vec
import re

with open('../class/data/三国演义.txt') as f:
    document = f.read()
    document = re.sub('[，。？！：；、“”]+', ' ', document)  # 去标点
    document_cut = jieba.cut(document)  # 结巴分词
    result = ' '.join(document_cut)
    with open('1.txt', 'w') as f2:
        f2.write(result)

sentences = word2vec.LineSentence('1.txt')
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3)

s1 = model.wv.most_similar('曹操')
s2 = model.wv.most_similar('玄德')


def show(s, name):
    print(name + ':', end=' ')
    for i in s:
        print(i[0], end=' ')
    print()


show(s1, '曹操')
show(s2, '玄德')
