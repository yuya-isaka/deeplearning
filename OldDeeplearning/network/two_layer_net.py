import sys, os
sys.path.append(os.pardir)
import numpy as np  
from OldDeeplearning.common.function import sigmoid, softmax, cross_entropy_error
# from common.gradient import numerical_gradient
from OldDeeplearning.common.layers import Relu, Affine, SoftmaxWithLoss
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x): # 引数xは画像データ、出力はニューラルネットワークのだした値
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)

        # layersに入ってるのを出してforward処理を順番にしていく
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t): # 引数xは画像データ、引数tは正解ラベル、出力は誤差
        y = self.predict(x)

        # ここでsoftmax関数も一緒にやる
        return self.lastLayer.forward(y, t) # predictの結果と正解ラベルを使う

    def accuracy(self, x, t): # 引数xは画像データ、引数tは正解ラベル
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    # def numerical_gradient(self, x, t): # 引数xは画像データ、引数tは正解ラベル
    #     loss_W = lambda W: self.loss(x, t)

    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    #     grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    #     grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    #     return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

