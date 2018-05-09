
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import zeros
from numpy import log
import numpy as np
from scipy.linalg import svd
import jieba

#停用词，在选择关键词的时候需要剔除
stopwords = ['的','是','不']
#标点符号，同样需要删除
ignorechars = ''',:'!，。：'''


class LSA(object):

    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0

    def parse(self, doc):
        # 采用结巴分词
        words = [i for i in jieba.cut_for_search(doc, HMM=True)]
        for w in words:
            w = w.lower().strip(ignorechars).strip()
            
            if w in self.stopwords or not w:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1
    #用来计算TFIDF
    def TFIDF(self):
        WordsPerDoc = self.A.sum(axis=0)
        DocsPerWord = np.asarray(self.A > 0, 'i').sum(axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                if self.A[i, j]:
                    self.A[i, j] = (self.A[i, j] / WordsPerDoc[j]) * \
                        log(float(cols) / DocsPerWord[i])

    def SVD(self):
        self.U, self.S, self.Vt = svd(self.A)

    def getA(self):
        # print(self.A)
        return self.A

    def getU(self):
        # print(self.U)
        return self.U

    def getS(self):
        # print(self.S)
        return self.S

    def getV(self):
        # print(self.Vt)
        return self.Vt

    def get_keys(self):
        return self.keys

    def get_wdict(self):
        return self.wdict

#把A矩阵经过SVD分解得到的三个矩阵，保留前maintain_dimen维，重新得到A’矩阵，
#实现了除噪的效果。
def make_new_matrix(U, S, V, maintain_dimen):
    m = U.shape[0]
    n = V.shape[0]
    U_new = zeros((m, maintain_dimen))
    S_new = zeros((maintain_dimen, maintain_dimen))
    V_new = zeros((maintain_dimen, n))
    for i in range(maintain_dimen):
        U_new[:, i] = U[:, i]
        S_new[i][i] = S[i]
        V_new[i, :] = V[i, :]

    A_new = np.dot(np.dot(U_new, S_new), V_new)
    return A_new

#从文件中读取中文标题
titles = []
with open('Chinese_titles_romance.txt', encoding='utf-8') as title_file:
    for line in title_file:
        titles.append(line.strip())
titles_num = len(titles)

#构建LSA模型
mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)
mylsa.build()
A = mylsa.getA()
mylsa.TFIDF()

#查看重构前的矩阵
print('*****************\nThe naive A matrix:')
print(A)

#执行SVD分解
mylsa.SVD()
U = mylsa.getU()
S = mylsa.getS()
V = mylsa.getV()

#查看词频统计
print("*****************\nWord_dictionary:")
w_dict = mylsa.get_wdict()
print(w_dict)

#查看key_words
print("*****************\nKey words:")
key_words = mylsa.get_keys()
print(key_words)

#重构A矩阵，保留前2维度
maintain_dimen = 2
A_new = make_new_matrix(U, S, V, maintain_dimen)
print('*****************\nThe new A matrix:')
print(A_new)
