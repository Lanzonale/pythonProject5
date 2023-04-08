#此为保存的模型，需配合四个储存参数的csv表格使用
import numpy as np
import pandas as pd#用以读取表格
from set import sigmoid
from set import softmax

P = pd.read_csv('./W1.csv')
Q = pd.read_csv('./W2.csv')
p = pd.read_csv('./b1.csv')
q = pd.read_csv('./b2.csv')

P = P.values
Q = Q.values
p = p.values[:, 1]#由于以csv储存，需取array值，去掉前方目录数
q = q.values[:, 1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)#防止溢出
    return np.exp(x) / np.sum(np.exp(x))

def classfuc(x):#进行分类
    ans = softmax(np.dot(sigmoid((np.dot(x, P) + p)), Q) + q)
    return ans



