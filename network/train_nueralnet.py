import sys, os
sys.path.append(os.pardir)
import numpy as np 
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from network.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True) #mnistのデータを正規化on,一次元化on,one-hot表現off

train_loss_list = [] #損失の記録リスト
train_acc_list = [] #訓練データの正解率リスト
test_acc_list = [] #テストデータの正解率リスト

# ハイパーパラメータ
iters_num = 10000 #学習回数
batch_size = 100 #ミニバッチの大きさ
learning_rate = 0.1 #学習率

train_size = x_train.shape[0] #訓練データの数

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #二層のニューラルネットワークを生成

iter_per_epoch = max(train_size / batch_size, 1) #何回周期で記録するか

#指定した回数ミニバッチを取り出して勾配求めてパラメータを更新する
for i in range(iters_num):

    #ミニバッチを取り出す
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #ミニバッチの勾配を求める
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    #勾配を基にに繰り返しパラメータl；。を更新（勾配降下法）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #損失を記録する
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='訓練 認識精度')
# plt.plot(x, test_acc_list, label='テスト 認識精度', linestyle='--')
# plt.xlabel("エポック")
# plt.ylabel("精度")
# plt.ylim(0, 1.0) # yの出力範囲指定
# plt.legend(loc='lower right') # ラベルの表示位置を決められる
# plt.show()

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()