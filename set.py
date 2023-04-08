#构建二层网路的set文件，用以导入至main，其中包含激活函数、各层及数学操作
import numpy as np

def sigmoid(x):#激活函数
    return 1 / (1 + np.exp(-x))

def softmax(x):#输出归一化
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)#防止溢出
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid_grad(x):
        return (1.0 - sigmoid(x)) * sigmoid(x)

def numerical_gradient(f, x):#计算梯度
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

        return grad


def cross_entropy_error(self,y, t,lamb):#loss的计算
    # 交叉熵函数，其中标签使用one-hot-vector
    W1, W2 = self.params['W1'], self.params['W2']
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    cs1=-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # l2正则化项
    l2_cs = lamb * (np.sum(np.square(W1)) + np.sum(np.square(W2))) / (2*batch_size)
    return cs1+l2_cs

class Net:#整合功能
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化网络
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):#前向传播
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y


    def loss(self, x, t,lamb):
        y = self.predict(x)
        loss = cross_entropy_error(self,y, t,lamb)

        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t,lamb):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    #反向传播计算梯度
    def gradient(self, x, t,lamb):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]


        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)


        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)+lamb*W2/batch_num
        grads['b2'] = np.sum(dy, axis=0)



        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)+lamb*W1/batch_num
        grads['b1'] = np.sum(da1, axis=0)

        return grads

