import numpy as np
import pandas as pd#用以把模型储存至本地
from Mnist import load_mnist#读取mnist数据集
from set import Net#二层网路层
import matplotlib.pyplot as plt#输出iteration&accuracy
from Model import classfuc #classfuc为另存在model的结果模型，最后调用测试一次模型

#导出数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

#设置超参数
iters_num = 20000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1
lamb = 0.1
iter_per_epoch = max(train_size / batch_size, 1)
#神经网路大小
network = Net(input_size=784, hidden_size=50, output_size=10)
#SGD，随机选取minibatch迭代
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    batch2_mask = np.random.choice(test_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    x_batch2 = x_test[batch2_mask]
    t_batch2 = t_test[batch2_mask]
    #train的梯度
    grad = network.gradient(x_batch, t_batch, lamb)
    #test的梯度
    grad = network.gradient(x_batch2, t_batch2, lamb)

    #更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        if (i % 100 == 0):
            LR = learning_rate * (i / 10000)
        network.params[key] -= LR * grad[key]
        if (i == iters_num - 1):
            #导出最后的系数，用以构建模型
            P = network.params['W1']
            W1=pd.DataFrame(P)
            Q = network.params['W2']
            W2=pd.DataFrame(Q)
            p = network.params['b1']
            b1=pd.DataFrame(p)
            q = network.params['b2']
            b2=pd.DataFrame(q)

    #记录训练和测试的loss
    loss = network.loss(x_batch, t_batch, lamb)
    loss2 = network.loss(x_batch2, t_batch2, lamb)
    train_loss_list.append(loss)
    test_loss_list.append(loss2)
    #记录训练和测试的accuracy
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc&test acc =" + str(train_acc) + "&" + str(test_acc))

# 画出测试和训练的损失函数
x1 = np.arange(len(train_loss_list))
ax1 = plt.subplot(121)
plt.plot(x1, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")

x2 = np.arange(len(test_loss_list))
ax1 = plt.subplot(122)
plt.plot(x2, test_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")

# 训练精度，测试精度随着epoch的变化
markers = {'train': 'o', 'test': 's'}
x2 = np.arange(len(train_acc_list))
ax2 = plt.subplot(122)
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc', linestyle='-')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')#在右下角显示图标
plt.show()

#——————主函数到以上结束，下面为测试classfuc可行性的代码——————
f=x_test[6683,:]#随机选定test一行数据进行分类
v = classfuc(f)#放入模型
print(v)#显示分类结果
u=t_test[6683,:]#读取对应的标签
print(u)
#print结果显示v最大值确实对应标签1，分类成功，结果如下
#[1.66132426e-05 9.77210622e-01 6.96992823e-03 5.67293630e-04
 #8.11822384e-04 1.30637788e-03 8.42151456e-04 3.46691184e-03
 #8.47151832e-03 3.36760760e-04]
#[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]




