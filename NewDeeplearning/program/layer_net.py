# coding: utf-8
# aaaa
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.layers_test import *
from collections import OrderedDict
import numba
from copy import deepcopy
import random

from program.crossbar import Crossbar

# メモリスタニューラルネットワークのクラス．各種パラメータを保持する．
# 計算は，layers_test.pyに保存されているクラスファイルを利用して行う．
# この辺の計算の詳細はオライリーの本（5章，特に5.6節）を参考に実現されるため，そちらを参照していただきたい．
class MNN:
    def __init__(self, size, activation_function="Relu", batch_norm=False, mu=0, std=0, row_test_size=1, col_test_size=1, soft_fault_rate=0, hard_fault=False, baseline=False, prop=False, seed1=0, seed2=0, fault=False):
        # 重みの初期化
        # 重みの初期値はとりあえずXavierの1/root(n)で。
        self.size = size
        self.params = {}
        self.con_p = {} # +用のメモリスタ
        self.con_n = {} # -用のメモリスタ
        self.crossbar = {}
        self.nameW = []
        self.batch_norm = batch_norm
        # ITCにのってたデータの標準偏差部分を改良
        self.end_mu  = mu
        self.end_std = std
        self.end_p = {}
        self.end_n = {}
        # soft_faultが発生するかしないか．fault freeの実験のときのみ，False．
        self.soft_fault_rate = soft_fault_rate
        # hard_faultが発生するかしないか．fault freeの実験のときのみ，False．
        self.hard_fault = hard_fault

        self.row_test_size = row_test_size
        self.col_test_size = col_test_size

        self.prop = prop
        self.soft_fault_rate = soft_fault_rate

        self.seed1 = seed1
        self.seed2 = seed2

        self.fault = fault
        self.activation_function = activation_function

        self.baseline = baseline

        for i in range(len(self.size)-1):
            # 重みパラメータ
            np.random.seed(self.seed1+i)
            self.params['W'+str(i+1)] = np.round(np.random.randn(int(size[i]), int(size[i+1])) / np.sqrt(int(size[i])), decimals=6)
            # self.params['W'+str(i+1)] = np.round(np.random.randint(-100, 100, (int(size[i]), int(size[i+1])))) #/ np.sqrt(int(size[i])), decimals=6)

            self.con_p['W'+str(i+1)]  = deepcopy(self.params['W'+str(i+1)])
            self.con_p['W'+str(i+1)]  = np.where(self.con_p['W'+str(i+1)] <= 0, 0, self.con_p['W'+str(i+1)])
            self.con_n['W'+str(i+1)]  = deepcopy(self.params['W'+str(i+1)])
            self.con_n['W'+str(i+1)]  = np.where(self.con_n['W'+str(i+1)] >= 0, 0, self.con_n['W'+str(i+1)])
            self.con_n['W'+str(i+1)]  *= -1

            # self.end_p['W'+str(i+1)] = np.round(np.random.normal(self.end_mu, self.end_std, (int(size[i]), int(size[i+1]))), decimals=0)
            # self.end_n['W'+str(i+1)] = np.round(np.random.normal(self.end_mu, self.end_std, (int(size[i]), int(size[i+1]))), decimals=0)

            self.nameW.append("W"+str(i+1))

            self.crossbar["P"+str(i+1)] = Crossbar(W=self.con_p["W"+str(i+1)], row_size=self.con_p["W"+str(i+1)].shape[0], column_size=self.con_p["W"+str(i+1)].shape[1], row_test_size=self.row_test_size, column_test_size=self.col_test_size, fault_rate=0, prop=self.prop, seed=self.seed2+i)
            self.crossbar["N"+str(i+1)] = Crossbar(W=self.con_n["W"+str(i+1)], row_size=self.con_n["W"+str(i+1)].shape[0], column_size=self.con_n["W"+str(i+1)].shape[1], row_test_size=self.row_test_size, column_test_size=self.col_test_size, fault_rate=0, prop=self.prop, seed=self.seed2+i+100)

        # レイヤの生成
        self.layers = OrderedDict()
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)] = Affine_MEM(self.params['W'+str(i+1)], self.con_p['W'+str(i+1)], self.con_n['W'+str(i+1)], self.end_mu, self.end_std, fault=fault, prop=self.prop, soft_fault_rate=self.soft_fault_rate, row_test_size=self.row_test_size, column_test_size=self.col_test_size, baseline=self.baseline, seed=self.seed1)
            self.layers['Affine'+str(i+1)].initialize_fault_map_and_endurance(self.crossbar["P"+str(i+1)].row_check_sum, self.crossbar["P"+str(i+1)].column_check_sum, self.crossbar["N"+str(i+1)].row_check_sum, self.crossbar["N"+str(i+1)].column_check_sum)
            if (i != len(size)-2):
                if self.batch_norm:
                    self.params['Gamma' + str(i+1)] = np.ones(int(size[i+1]))
                    self.params['Beta' + str(i+1)] = np.zeros(int(size[i+1]))
                    self.layers['BatchNorm' + str(i+1)] = BatchNormalization(self.params['Gamma' + str(i+1)], self.params['Beta' + str(i+1)])
                if self.activation_function == "Sigmoid":
                    self.layers[self.activation_function+str(i+1)] = Sigmoid()
                elif self.activation_function == "Relu":
                    self.layers[self.activation_function+str(i+1)] = Relu()
        self.lastLayer = SoftmaxWithLoss()

    # メモリスタに過渡故障を注入
    def soft_fault_apply(self):
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)].soft_fault_injection()

    # メモリスタに永久故障を注入
    def hard_fault_apply(self):
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)].reduce_endurance()
            self.layers['Affine'+str(i+1)].hard_fault_injection()

    # メモリスタに重みの値（故障値込み）をセット
    def crossbar_weight_set(self):
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)].set_weight()

    # 過渡故障と永久故障のマップを結合
    def combine_fault(self):
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)].combine_fault_map()
            self.layers['Affine'+str(i+1)].combine_fault_value()

    # メモリスタニューラルネットワークに対してテストを実行
    def test_apply(self):
        for i in range(len(self.size)-1):

            self.crossbar["P"+str(i+1)].initialize()
            self.crossbar["N"+str(i+1)].initialize()

            self.crossbar["P"+str(i+1)].set_check_sum(self.layers["Affine"+str(i+1)].Cp)
            self.crossbar["N"+str(i+1)].set_check_sum(self.layers["Affine"+str(i+1)].Cn)

            self.crossbar["P"+str(i+1)].set_fault_map(self.layers["Affine"+str(i+1)].combine_fault_value_p, self.layers["Affine"+str(i+1)].combine_fault_locate_p, self.layers["Affine"+str(i+1)].combine_fault_value_row_check_sum_p, self.layers["Affine"+str(i+1)].combine_fault_locate_row_check_sum_p, self.layers["Affine"+str(i+1)].combine_fault_value_column_check_sum_p, self.layers["Affine"+str(i+1)].combine_fault_locate_column_check_sum_p)
            self.crossbar["N"+str(i+1)].set_fault_map(self.layers["Affine"+str(i+1)].combine_fault_value_n, self.layers["Affine"+str(i+1)].combine_fault_locate_n, self.layers["Affine"+str(i+1)].combine_fault_value_row_check_sum_n, self.layers["Affine"+str(i+1)].combine_fault_locate_row_check_sum_n, self.layers["Affine"+str(i+1)].combine_fault_value_column_check_sum_n, self.layers["Affine"+str(i+1)].combine_fault_locate_column_check_sum_n)

            self.crossbar["P"+str(i+1)].fault_injection()
            self.crossbar["N"+str(i+1)].fault_injection()

            self.crossbar["P"+str(i+1)].test()
            self.crossbar["N"+str(i+1)].test()

    # テストによって故障を修復した結果をAffine_MEMのクラスで保存されている重み行列に適用する
    def correction_apply(self):
        for i in range(len(self.size)-1):
            self.layers["Affine"+str(i+1)].correction_map(self.crossbar["P"+str(i+1)].new_fault_map, self.crossbar["P"+str(i+1)].new_fault_map_row_check_sum, self.crossbar["P"+str(i+1)].new_fault_map_column_check_sum, self.crossbar["N"+str(i+1)].new_fault_map, self.crossbar["N"+str(i+1)].new_fault_map_row_check_sum, self.crossbar["N"+str(i+1)].new_fault_map_column_check_sum)

    def get_fault_rate(self):

        sum_p = 0
        sum_n = 0

        for i in range(len(self.size)-1):
            fault_rate_p = self.crossbar["P"+str(i+1)].cell_fault_rate
            fault_rate_n = self.crossbar["N"+str(i+1)].cell_fault_rate

            sum_p += fault_rate_p
            sum_n += fault_rate_n

        sum_p = sum_p / (len(self.size) - 1)
        sum_n = sum_n / (len(self.size) - 1)

        return sum_p, sum_n

    def w_fit_high(self):
        # wの重みを1〜0に正規化(そもそもこれであっているのか…)
        for key in (self.nameW):
            self.params[key][self.params[key] < -1] = -1
            self.params[key][self.params[key] > 1] = 1

    # ここは中間層のノードの出力値を求めてるだけです。
    # ただ、for文ぐるぐる回すのは時間かかるので、npのdot積演算による足し算を利用しています

    @numba.jit
    def predict(self, x, train_flg=False, acc=False):
        for key, layer in self.layers.items():
            if acc == True:
                if "Dropout" in key or "BatchNorm" in key:
                    x = layer.forward(x, train_flg)
                else:
                    #print("key is : "+str(key))
                    x = layer.forward(x)
            else:
                if "Dropout" in key or "BatchNorm" in key:
                    x = layer.forward(x, train_flg)
                else:
                    #print(self.layers.items())
                    x = layer.forward(x)
                    #print("ok"+str(key))
        return x

    # x:入力データ, t:教師データ
    @numba.jit
    def loss(self, x, t, train_flg=False):

        y = self.predict(x, train_flg)
        norm = 0

        return self.lastLayer.forward(y, t) #+ norm

    # 識別率を測定
    @numba.jit
    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False, acc=True)
        z = deepcopy(y)
        y = numpy.argmax(y, axis=1)
        if t.ndim != 1 : t = numpy.argmax(t, axis=1)

        accuracy = np.sum(np.asarray(y) == np.asarray(t)) / np.float(x.shape[0])

        cross_error = self.lastLayer.forward(z, t)

        return accuracy, cross_error

    # x:入力データ, t:教師データ
    # g = 0: normal, g = 1: glasso
    @numba.jit
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)
        # print("loss ok")

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i in range (len(self.nameW)):
            grads["W"+str(i+1)] = self.layers['Affine'+str(i+1)].dW
            if self.batch_norm and i+1 != len(self.nameW):
                grads['Gamma' + str(i+1)] = self.layers["BatchNorm" + str(i+1)].dgamma
                grads['Beta' + str(i+1)]  = self.layers["BatchNorm" + str(i+1)].dbeta

        return grads

    def affine_update(self):
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)].W = self.params['W'+str(i+1)]

    def affine_grads_update(self, grads):
        for i in range(len(self.size)-1):
            self.layers['Affine'+str(i+1)].dW = grads['W'+str(i+1)]

