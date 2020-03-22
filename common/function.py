import numpy as np 


#活性化関数、線形な入力を非線形にする、表現力の拡張
def step_function(x):
    #y = x > 0
    #return y.astype(np.int)
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)



#出力関数、学習で使うことが多い、推論時はそのまま値を出す。
def identify_function(x):
    return x


def softmax(x):
    if x.ndim == 2:
        x = x.T 
        x = x - np.max(x, axis=0) # オーバーフロー対策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


#損失関数、パラメータを更新する際の指標
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# numpy配列に対応していない
def _cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error(y, t):
    if y.ndim == 1: #バッチデータじゃないとき
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size: #one-hotベクトルの時、対応のインデックスに変更
        t = t.argmax(axis=1)

    batch_size = y.shape[0] #ニューラルネットワークの出力yに、何個の情報があるか（行を取り出す）
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #yの何行目のどのインデックスをみるか調べて、それのlogをとる。そこの値が大きいとlogは大きくなり損失が大きいということになる大きいということになる。


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad