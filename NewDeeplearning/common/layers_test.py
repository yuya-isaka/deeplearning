# coding: utf-8
import math
import numpy as np
from common.functions import *
from common.util import im2col, col2im
import numba
from copy import deepcopy
import bisect
import numpy 

#np.set_printoptions(np.inf)

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def forward_acc(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def forward_acc(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Tanh:
    def __init__(self):
        self.out = None

    @numba.jit
    def forward(self,x):
        out = tanh(x)
        self.out = out
        return out 

    @numba.jit
    def forward_acc(self,x):
        out = tanh(x)
        self.out = out
        return out 

    def forward_AE(self,x):
        out = tanh(x)
        self.out = out
        return out 

    @numba.jit
    def backward(self,dout):
        #dx = 1.0 - self.out**2
        dx = (1.0 - self.out**2) * dout
        return dx

class Affine_MEM:
    def __init__(self, W, Cp, Cn, end_mu, end_std, fault, prop, soft_fault_rate, row_test_size, column_test_size, baseline, seed):
        self.W = deepcopy(W)

        # 過渡故障の割合
        self.soft_fault_rate = soft_fault_rate

        # テストブロックの行/列サイズ
        self.row_test_size = row_test_size
        self.column_test_size = column_test_size

        # メモリスタの寿命の平均と標準偏差
        # メモリスタ単体の寿命は，基本的に正規分布に従います
        self.end_mu  = end_mu
        self.end_std = end_std

        # メモリスタのコンダクタンス値
        # 正負用に2つある
        self.Cp = Cp
        self.Cn = Cn

        # メモリスタニューラルネットワークに対する入力ベクトル
        self.x = None
        self.original_x_shape = None

        # ナニコレ
        self.dW = None
        self.db = None

        # fault：故障の有無，prop：提案手法かどうか，baseline：実験対象がbaselineかどうか
        self.fault    = fault
        self.prop     = prop
        self.baseline = baseline

        # 過渡故障，永久故障，寿命の乱数シードの設定．適当でOK．
        self.soft_seed = seed
        self.hard_seed = seed*100
        self.end_seed  = seed*10000

        self.per_column = 0
        self.per_row    = 0

        # 正のクロスバーアレイ中のメモリスタの寿命
        self.end_P    = None
        self.end_rowP = None
        self.end_colP = None

        # 負のクロスバーアレイ中のメモリスタの寿命
        self.end_N    = None
        self.end_rowN = None
        self.end_colN = None

        self.end_mask    = None
        self.end_rowmask = None
        self.end_colmask = None

        # self.end_p['W'+str(i+1)] = np.round(np.random.normal(self.end_mu, self.end_std, (int(size[i]), int(size[i+1]))), decimals=0)
        # self.end_n['W'+str(i+1)] = np.round(np.random.normal(self.end_mu, self.end_std, (int(size[i]), int(size[i+1]))), decimals=0)

        # 正の値用クロスバーアレイの永久故障(hard故障)の故障位置
        # "1"の場所が故障有り，"0"の場所が故障なしとして保存
        self.hard_fault_locate_p                  = None
        self.hard_fault_locate_row_check_sum_p    = None
        self.hard_fault_locate_column_check_sum_p = None

        # 正の値用クロスバーアレイの永久故障の故障値を保存
        # hard_fault_locate_pが"1"に対応する重み行列Wをhard_fault_value_pで更新する
        self.hard_fault_value_p                   = None
        self.hard_fault_value_row_check_sum_p     = None
        self.hard_fault_value_column_check_sum_p  = None

        # 負の値用クロスバーアレイの永久故障(hard故障)の故障位置
        # "1"の場所が故障有り，"0"の場所が故障なしとして保存
        self.hard_fault_locate_n                  = None
        self.hard_fault_locate_row_check_sum_n    = None
        self.hard_fault_locate_column_check_sum_n = None

        # 負の値用クロスバーアレイの永久故障の故障値を保存
        # hard_fault_locate_nが"1"に対応する重み行列Wをhard_fault_value_nで更新する
        self.hard_fault_value_n                   = None
        self.hard_fault_value_row_check_sum_n     = None
        self.hard_fault_value_column_check_sum_n  = None

        # 正の値用クロスバーアレイの過渡故障(soft故障)の故障位置
        # "1"の場所が故障有り，"0"の場所が故障なしとして保存
        self.soft_fault_locate_p                  = None
        self.soft_fault_locate_row_check_sum_p    = None
        self.soft_fault_locate_column_check_sum_p = None

        # 正の値用クロスバーアレイの過渡故障の故障値を保存
        # soft_fault_locate_pが"1"に対応する重み行列Wをhard_fault_value_pで更新する
        self.soft_fault_value_p                   = None
        self.soft_fault_value_row_check_sum_p     = None
        self.soft_fault_value_column_check_sum_p  = None

        # 負の値用クロスバーアレイの過渡故障(hard故障)の故障位置
        # "1"の場所が故障有り，"0"の場所が故障なしとして保存
        self.soft_fault_locate_n                  = None
        self.soft_fault_locate_row_check_sum_n    = None
        self.soft_fault_locate_column_check_sum_n = None

        # 負の値用クロスバーアレイの過渡故障の故障値を保存
        # soft_fault_locate_nが"1"に対応する重み行列Wをsoft_fault_value_nで更新する
        self.soft_fault_value_n                   = None
        self.soft_fault_value_row_check_sum_n     = None
        self.soft_fault_value_column_check_sum_n  = None

        # 正負の永久故障箇所をまとめたもの
        self.combine_fault_locate_p                  = None
        self.combine_fault_locate_row_check_sum_p    = None
        self.combine_fault_locate_column_check_sum_p = None

        # 正負の永久故障の値をまとめたもの
        self.combine_fault_value_p                   = None
        self.combine_fault_value_row_check_sum_p     = None
        self.combine_fault_value_column_check_sum_p  = None

        # 正負の永久故障箇所をまとめたもの
        self.combine_fault_locate_n                  = None
        self.combine_fault_locate_row_check_sum_n    = None
        self.combine_fault_locate_column_check_sum_n = None

        self.combine_fault_value_n                   = None
        self.combine_fault_value_row_check_sum_n     = None
        self.combine_fault_value_column_check_sum_n  = None

        self.sigma_rate = 1.0
        self.dis_log = True

        self.seed = seed

    def set_weight(self):

        self.Cp = np.where(self.W <= 0, 0, self.W)

        self.Cn = np.where(self.W >= 0, 0, self.W)
        self.Cn = self.Cn * -1

    def reduce_endurance(self):

        self.end_P    -= 1
        self.end_rowP -= 1
        self.end_colP -= 1

        self.end_N    -= 1
        self.end_rowN -= 1
        self.end_colN -= 1

    def hard_fault_injection(self):

        self.hard_fault_locate_p                  = np.where(self.end_P <= 0, 1, 0)
        self.hard_fault_locate_row_check_sum_p    = np.where(self.end_rowP <= 0, 1, 0)
        self.hard_fault_locate_column_check_sum_p = np.where(self.end_colP <= 0, 1, 0)

        self.hard_fault_locate_n                  = np.where(self.end_N <= 0, 1, 0)
        self.hard_fault_locate_row_check_sum_n    = np.where(self.end_rowN <= 0, 1, 0)
        self.hard_fault_locate_column_check_sum_n = np.where(self.end_colN <= 0, 1, 0)

    def initialize_fault_map_and_endurance(self, p_row, p_column, n_row, n_column):

        self.hard_fault_locate_p                  = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.hard_fault_locate_row_check_sum_p    = np.zeros((p_row.shape[0], p_row.shape[1]))
        self.hard_fault_locate_column_check_sum_p = np.zeros((p_column.shape[0], p_column.shape[1]))

        np.random.seed(self.hard_seed)

        self.hard_fault_value_p                   = np.random.randint(0, 2, (int(self.W.shape[0]), int(self.W.shape[1])))
        self.hard_fault_value_row_check_sum_p     = np.random.randint(0, 2, (int(p_row.shape[0]), int(p_row.shape[1])))
        self.hard_fault_value_column_check_sum_p  = np.random.randint(0, 2, (int(p_column.shape[0]), int(p_column.shape[1])))

        self.hard_fault_locate_n                  = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.hard_fault_locate_row_check_sum_n    = np.zeros((n_row.shape[0], n_row.shape[1]))
        self.hard_fault_locate_column_check_sum_n = np.zeros((n_column.shape[0], n_column.shape[1]))

        self.hard_seed += 1
        np.random.seed(self.hard_seed)

        self.hard_fault_value_n                  = np.random.randint(0, 2, (int(self.W.shape[0]),int(self.W.shape[1])))
        self.hard_fault_value_row_check_sum_n    = np.random.randint(0, 2, (int(n_row.shape[0]),int(n_row.shape[1])))
        self.hard_fault_value_column_check_sum_n = np.random.randint(0, 2, (int(n_column.shape[0]),int(n_column.shape[1])))

        self.soft_fault_locate_p                  = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.soft_fault_locate_row_check_sum_p    = np.zeros((p_row.shape[0], p_row.shape[1]))
        self.soft_fault_locate_column_check_sum_p = np.zeros((p_column.shape[0], p_column.shape[1]))

        self.soft_fault_value_p                   = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.soft_fault_value_row_check_sum_p     = np.zeros((p_row.shape[0], p_row.shape[1]))
        self.soft_fault_value_column_check_sum_p  = np.zeros((p_column.shape[0], p_column.shape[1]))

        self.soft_fault_locate_n                  = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.soft_fault_locate_row_check_sum_n    = np.zeros((n_row.shape[0], n_row.shape[1]))
        self.soft_fault_locate_column_check_sum_n = np.zeros((n_column.shape[0], n_column.shape[1]))

        self.soft_fault_value_n                   = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.soft_fault_value_row_check_sum_n     = np.zeros((n_row.shape[0], n_row.shape[1]))
        self.soft_fault_value_column_check_sum_n  = np.zeros((n_column.shape[0], n_column.shape[1]))

        self.hard_seed += 1
        np.random.seed(self.hard_seed)
        self.end_P    = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.hard_fault_value_p.shape[0]), int(self.hard_fault_value_p.shape[1]))), decimals=0)

        self.hard_seed += 1
        np.random.seed(self.hard_seed)
        self.end_rowP = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.hard_fault_value_row_check_sum_p.shape[0]), int(self.hard_fault_value_row_check_sum_p.shape[1]))), decimals=0)

        self.hard_seed += 1
        np.random.seed(self.hard_seed)
        self.end_colP = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.hard_fault_value_column_check_sum_p.shape[0]), int(self.hard_fault_value_column_check_sum_p.shape[1]))), decimals=0)

        self.hard_seed += 1
        np.random.seed(self.hard_seed)
        self.end_N    = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.hard_fault_value_n.shape[0]), int(self.hard_fault_value_n.shape[1]))), decimals=0)

        self.hard_seed += 1
        np.random.seed(self.hard_seed)
        self.end_rowN = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.hard_fault_value_row_check_sum_n.shape[0]), int(self.hard_fault_value_row_check_sum_n.shape[1]))), decimals=0)

        self.hard_seed += 1
        np.random.seed(self.hard_seed)
        self.end_colN = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.hard_fault_value_column_check_sum_n.shape[0]), int(self.hard_fault_value_column_check_sum_n.shape[1]))), decimals=0)

        # self.end_p['W'+str(i+1)] = np.round(np.random.normal(self.end_mu, self.end_std, (int(size[i]), int(size[i+1]))), decimals=0)
        # self.end_n['W'+str(i+1)] = np.round(np.random.normal(self.end_mu, self.end_std, (int(size[i]), int(size[i+1]))), decimals=0)

    def G_variation(self, w, sigma_rate, dis_log):
        
        if sigma_rate == 0:
            return w

        max_con = 1/10000
        min_con = 1/1000000

        G = w * (max_con - min_con) + min_con
        R = 1 / G

        #print("w is : "+str(w.shape))

        if dis_log == True:
            mux    = R #* np.log(10)
            #std_var = (mux - Rmin) / (Rmax - Rmin) * (1 - 0.25) + 0.25

            sigmax = sigma_rate * np.copy(mux) #* std_var

            sigmai = np.sqrt(np.log((1 + (sigmax * sigmax) / (mux * mux))))
            mui    = np.log(mux) - sigmai * sigmai / 2

            #Rv = np.zeros((w.shape[0], w.shape[1]))
            #for i in range(w.shape[0]):
            #    for j in range(w.shape[1]):
                    #ran = np.random.randint(1000000)
                    #np.random.seed(ran)
            #        Rv[i][j] = np.random.normal(mui[i][j], sigmai[i][j])

            #print("mui is : "+str(mui.shape))
            #print("sigmai is : "+str(sigmai.shape))

            #Rv = np.asarray(numpy.random.lognormal(np.asnumpy(mui), np.asnumpy(sigmai), (w.shape[0], w.shape[1])))
            Rv = np.random.normal(mui, sigmai, (w.shape[0], w.shape[1]))
            Rv = np.exp(Rv)
            #Rv = np.random.lognormal(mui, sigmai, (w.shape[0], w.shape[1]))
            #Rv = np.zeros((w.shape[0], w.shape[1]))
            #for i in range(w.shape[0]):
            #    for j in range(w.shape[1]):
            #        Rv[i][j] = np.random.lognormal(mui[i][j], sigmai[i][j])
            Gv = 1 / Rv
            wv = (Gv - min_con) / (max_con - min_con)
            #print("wv is :"+str(wv))

            return(wv)

        else:
            mux     = R 
            #std_var = (mux - Rmin) / (Rmax - Rmin) * (1 - 0.25) + 0.25

            sigmax = sigma_rate * np.copy(mux) #* std_var

            Rv = np.random.normal(mux, sigmax, (w.shape[0], w.shape[1]))
            Gv = 1 / Rv

            wv = (Gv - min_con) / (max_con - min_con)
            #print("wv is :"+str(wv))

            return(wv)

    def soft_fault_injection(self):

        np.random.seed(self.soft_seed)
        self.soft_fault_locate_p                  = np.random.binomial(n=1, p=self.soft_fault_rate, size=((self.soft_fault_locate_p.shape[0], self.soft_fault_locate_p.shape[1])))
        self.soft_seed += 1
        np.random.seed(self.soft_seed)
        self.soft_fault_locate_row_check_sum_p    = np.random.binomial(n=1, p=self.soft_fault_rate, size=((self.soft_fault_locate_row_check_sum_p.shape[0], self.soft_fault_locate_row_check_sum_p.shape[1])))
        self.soft_seed += 1
        np.random.seed(self.soft_seed)
        self.soft_fault_locate_column_check_sum_p = np.random.binomial(n=1, p=self.soft_fault_rate, size=((self.soft_fault_locate_column_check_sum_p.shape[0], self.soft_fault_locate_column_check_sum_p.shape[1])))

        self.soft_fault_value_p                  = np.random.rand(self.soft_fault_locate_p.shape[0], self.soft_fault_locate_p.shape[1])
        self.soft_fault_value_row_check_sum_p    = np.random.rand(self.soft_fault_locate_row_check_sum_p.shape[0], self.soft_fault_locate_row_check_sum_p.shape[1])
        self.soft_fault_value_column_check_sum_p = np.random.rand(self.soft_fault_locate_column_check_sum_p.shape[0], self.soft_fault_locate_column_check_sum_p.shape[1])
        #self.soft_fault_value_p                  = self.G_variation(self.Cp, self.sigma_rate, self.dis_log)
        #self.soft_fault_value_row_check_sum_p    = self.G_variation(self.Cp, self.sigma_rate, self.dis_log)
        #self.soft_fault_value_column_check_sum_p = self.G_variation(self.Cp, self.sigma_rate, self.dis_log)

        self.soft_seed += 1
        np.random.seed(self.soft_seed)
        self.soft_fault_locate_n                  = np.random.binomial(n=1, p=self.soft_fault_rate, size=((self.soft_fault_locate_n.shape[0], self.soft_fault_locate_n.shape[1])))
        self.soft_seed += 1
        np.random.seed(self.soft_seed)
        self.soft_fault_locate_row_check_sum_n    = np.random.binomial(n=1, p=self.soft_fault_rate, size=((self.soft_fault_locate_row_check_sum_n.shape[0], self.soft_fault_locate_row_check_sum_n.shape[1])))
        self.soft_seed += 1
        np.random.seed(self.soft_seed)
        self.soft_fault_locate_column_check_sum_n = np.random.binomial(n=1, p=self.soft_fault_rate, size=((self.soft_fault_locate_column_check_sum_n.shape[0], self.soft_fault_locate_column_check_sum_n.shape[1])))

        self.soft_fault_value_n                  = np.random.rand(self.soft_fault_locate_n.shape[0], self.soft_fault_locate_n.shape[1])
        self.soft_fault_value_row_check_sum_n    = np.random.rand(self.soft_fault_locate_row_check_sum_p.shape[0], self.soft_fault_locate_row_check_sum_p.shape[1])
        self.soft_fault_value_column_check_sum_n = np.random.rand(self.soft_fault_locate_column_check_sum_p.shape[0], self.soft_fault_locate_column_check_sum_p.shape[1])

        #self.soft_fault_value_n                  = self.G_variation(self.Cn, self.sigma_rate, self.dis_log)
        #self.soft_fault_value_row_check_sum_n    = self.G_variation(self.Cn, self.sigma_rate, self.dis_log)
        #self.soft_fault_value_column_check_sum_n = self.G_variation(self.Cn, self.sigma_rate, self.dis_log)

        self.soft_seed += 1

    def correction_map(self, correct_fault_locate_p, correct_fault_locate_row_check_sum_p, correct_fault_locate_column_check_sum_p, correct_fault_locate_n, correct_fault_locate_row_check_sum_n, correct_fault_locate_column_check_sum_n):

        self.soft_fault_locate_p                  = np.logical_and(self.soft_fault_locate_p, correct_fault_locate_p)
        self.hard_fault_locate_p                  = np.logical_and(self.hard_fault_locate_p, correct_fault_locate_p)

        self.soft_fault_locate_n                  = np.logical_and(self.soft_fault_locate_n, correct_fault_locate_n)
        self.hard_fault_locate_n                  = np.logical_and(self.hard_fault_locate_n, correct_fault_locate_n)

        self.soft_fault_locate_row_check_sum_p    = np.logical_and(self.soft_fault_locate_row_check_sum_p, correct_fault_locate_row_check_sum_p)
        self.hard_fault_locate_row_check_sum_p    = np.logical_and(self.hard_fault_locate_row_check_sum_p, correct_fault_locate_row_check_sum_p)

        self.soft_fault_locate_row_check_sum_n    = np.logical_and(self.soft_fault_locate_row_check_sum_n, correct_fault_locate_row_check_sum_n)
        self.hard_fault_locate_row_check_sum_n    = np.logical_and(self.hard_fault_locate_row_check_sum_n, correct_fault_locate_row_check_sum_n)

        self.soft_fault_locate_column_check_sum_p = np.logical_and(self.soft_fault_locate_column_check_sum_p, correct_fault_locate_column_check_sum_p)
        self.hard_fault_locate_column_check_sum_p = np.logical_and(self.hard_fault_locate_column_check_sum_p, correct_fault_locate_column_check_sum_p)

        self.soft_fault_locate_column_check_sum_n = np.logical_and(self.soft_fault_locate_column_check_sum_n, correct_fault_locate_column_check_sum_n)
        self.hard_fault_locate_column_check_sum_n = np.logical_and(self.hard_fault_locate_column_check_sum_n, correct_fault_locate_column_check_sum_n)

        self.combine_fault_map()

        # hard faultの寿命部分を1から生成し直す
        # マスク生成
        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_mask    = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.end_P.shape[0]), int(self.end_P.shape[1]))), decimals=0)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_rowmask = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.end_rowP.shape[0]), int(self.end_rowP.shape[1]))), decimals=0)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_colmask = np.round(np.random.normal(self.end_mu, self.end_std, (int(self.end_colP.shape[0]), int(self.end_colP.shape[1]))), decimals=0)

        # 寿命元に戻す

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_P     = np.where((self.end_P <= 0) & (self.hard_fault_locate_p == 0), self.end_mask, self.end_P)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_rowP  = np.where((self.end_rowP <= 0) & (self.hard_fault_locate_row_check_sum_p == 0), self.end_rowmask, self.end_rowP)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_colP  = np.where((self.end_colP <= 0) & (self.hard_fault_locate_column_check_sum_p == 0), self.end_colmask, self.end_colP)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_N     = np.where((self.end_N <= 0) & (self.hard_fault_locate_n == 0), self.end_mask, self.end_N)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_rowN  = np.where((self.end_rowN <= 0) & (self.hard_fault_locate_row_check_sum_n == 0), self.end_rowmask, self.end_rowN)

        self.end_seed += 1
        np.random.seed(self.end_seed)
        self.end_colN  = np.where((self.end_colN <= 0) & (self.hard_fault_locate_column_check_sum_n == 0), self.end_colmask, self.end_colN)

    def forward(self, x):
        # 入力データの加工. x = 8 x 8(64)の画像データ．バッチなので数は100，すなわち，64 x 100
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        if (self.fault==True):

            #self.combine_fault_map()
            #self.combine_fault_value()
            if (self.baseline == False):
                self.forward_error_correction()

            self.Cp = np.where(self.W <= 0, 0, self.W)

            self.Cn = np.where(self.W >= 0, 0, self.W)
            self.Cn = self.Cn * -1

            self.Cp = np.where(self.combine_fault_locate_p == 1, self.combine_fault_value_p, self.Cp)
            self.Cn = np.where(self.combine_fault_locate_n == 1, self.combine_fault_value_n, self.Cn)

            outP = np.dot(self.x, self.Cp)
            outN = np.dot(self.x, self.Cn)

            out = outP - outN

        else:
            out = np.dot(self.x, self.W)

        # print("out.shape is : "+str(out.shape))

        return out

    def combine_fault_map(self):

        self.combine_fault_locate_p                  = np.logical_or(self.soft_fault_locate_p, self.hard_fault_locate_p)
        self.combine_fault_locate_row_check_sum_p    = np.logical_or(self.soft_fault_locate_row_check_sum_p, self.hard_fault_locate_row_check_sum_p)
        self.combine_fault_locate_column_check_sum_p = np.logical_or(self.soft_fault_locate_column_check_sum_p, self.hard_fault_locate_column_check_sum_p)

        self.combine_fault_locate_n                  = np.logical_or(self.soft_fault_locate_n, self.hard_fault_locate_n)
        self.combine_fault_locate_row_check_sum_n    = np.logical_or(self.soft_fault_locate_row_check_sum_n, self.hard_fault_locate_row_check_sum_n)
        self.combine_fault_locate_column_check_sum_n = np.logical_or(self.soft_fault_locate_column_check_sum_n, self.hard_fault_locate_column_check_sum_n)

    def combine_fault_value(self):

        self.combine_fault_value_p                   = np.where(self.hard_fault_locate_p == 1, self.hard_fault_value_p, self.soft_fault_value_p)
        self.combine_fault_value_row_check_sum_p     = np.where(self.hard_fault_locate_row_check_sum_p == 1, self.hard_fault_value_row_check_sum_p, self.soft_fault_value_row_check_sum_p)
        self.combine_fault_value_column_check_sum_p  = np.where(self.hard_fault_locate_column_check_sum_p == 1, self.hard_fault_value_column_check_sum_p, self.soft_fault_value_column_check_sum_p)

        self.combine_fault_value_n                   = np.where(self.hard_fault_locate_n == 1, self.hard_fault_value_n, self.soft_fault_value_n)
        self.combine_fault_value_row_check_sum_n     = np.where(self.hard_fault_locate_row_check_sum_n == 1, self.hard_fault_value_row_check_sum_n, self.soft_fault_value_row_check_sum_n)
        self.combine_fault_value_column_check_sum_n  = np.where(self.hard_fault_locate_column_check_sum_n == 1, self.hard_fault_value_column_check_sum_n, self.soft_fault_value_column_check_sum_n)

    # 順方向伝播時の誤り訂正
    def forward_error_correction(self):

        self.per_column = int(np.floor(self.W.shape[1] / self.column_test_size))
        self.per_row    = int(np.floor(self.W.shape[0] / self.row_test_size))

        output_WP      = np.sum(self.combine_fault_locate_p, axis=0)
        #output_rowP    = np.sum(self.soft_fault_locate_row_check_sum_p, axis=0)
        output_columnP = np.sum(self.combine_fault_locate_column_check_sum_p, axis=0)

        output_WN      = np.sum(self.combine_fault_locate_n, axis=0)
        #output_rowN    = np.sum(self.soft_fault_locate_row_check_sum_n, axis=0)
        output_columnN = np.sum(self.combine_fault_locate_column_check_sum_n, axis=0)

        output_WP      = np.where(output_WP >= 1, 1, output_WP)
        #output_rowP    = np.where(output_rowP >= 1, 1, output_rowP)
        output_columnP = np.where(output_columnP >= 1, 1, output_columnP)

        output_WN      = np.where(output_WN >= 1, 1, output_WN)
        #output_rowN    = np.where(output_rowN >= 1, 1, output_rowN)
        output_columnN = np.where(output_columnN >= 1, 1, output_columnN)

        forward_error_rate = 0

        for i in range(self.per_column):

            if (np.sum(output_WP[int(i*self.column_test_size):int((i+1)*self.column_test_size)]) + np.sum(output_columnP[int(i*2):int((i+1)*2)])) <= 1:
                self.combine_fault_locate_p[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)] = 0
                forward_error_rate += 1

            if (np.sum(output_WN[int(i*self.column_test_size):int((i+1)*self.column_test_size)]) + np.sum(output_columnN[int(i*2):int((i+1)*2)])) <= 1:
                self.combine_fault_locate_n[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)] = 0
                forward_error_rate += 1

        # print("forward_error_rate is : "+str(forward_error_rate))

    # 誤差逆伝播時の誤り訂正
    def backward_error_correction(self):

        self.per_column = int(np.floor(self.W.shape[1] / self.column_test_size))
        self.per_row    = int(np.floor(self.W.shape[0] / self.row_test_size))

        output_WP      = np.sum(self.combine_fault_locate_p, axis=1)
        output_rowP    = np.sum(self.combine_fault_locate_row_check_sum_p, axis=1)
        #output_columnP = np.sum(self.soft_fault_locate_column_check_sum_p, axis=0)

        output_WN      = np.sum(self.combine_fault_locate_n, axis=1)
        output_rowN    = np.sum(self.combine_fault_locate_row_check_sum_n, axis=1)
        #output_columnN = np.sum(self.soft_fault_locate_column_check_sum_n, axis=0)

        output_WP      = np.where(output_WP >= 1, 1, output_WP)
        output_rowP    = np.where(output_rowP >= 1, 1, output_rowP)
        #output_columnP = np.where(output_columnP >= 1, 1, output_columnP)

        output_WN      = np.where(output_WN >= 1, 1, output_WN)
        output_rowN    = np.where(output_rowN >= 1, 1, output_rowN)
        #output_columnN = np.where(output_columnN >= 1, 1, output_columnN)

        backward_error_rate = 0

        for i in range(self.per_row):

            if (np.sum(output_WP[int(i*self.row_test_size):int((i+1)*self.row_test_size)]) + np.sum(output_rowP[int(i*self.row_test_size):int((i+1)*self.row_test_size)])) <= 1:
                self.combine_fault_locate_p[int(i*self.row_test_size):int((i+1)*self.row_test_size),:] = 0
                backward_error_rate += 1

            if (np.sum(output_WN[int(i*self.row_test_size):int((i+1)*self.row_test_size)]) + np.sum(output_rowN[int(i*self.row_test_size):int((i+1)*self.row_test_size)])) <= 1:
                self.combine_fault_locate_n[int(i*self.row_test_size):int((i+1)*self.row_test_size),:] = 0
                backward_error_rate += 1

        # print("backward_error_rate is : "+str(backward_error_rate))

    def forward_acc(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        # ここにeccの処理

        if self.binarize == True:
            out = np.dot(self.x, self.Wb)
        else:
            out = np.dot(self.x, self.W)

        return out

    def backward(self, dout):

        if (self.fault == True):

            if (self.prop == True):
                self.backward_error_correction()

            self.Cp = np.where(self.W <= 0, 0, self.W)

            self.Cn = np.where(self.W >= 0, 0, self.W)
            self.Cn = self.Cn * -1

            self.Cp = np.where(self.combine_fault_locate_p == 1, self.combine_fault_value_p, self.Cp)
            self.Cn = np.where(self.combine_fault_locate_n == 1, self.combine_fault_value_n, self.Cn)

            dx_p = np.dot(dout, self.Cp.T)
            dx_n = np.dot(dout, self.Cn.T)

            self.dW = np.dot(self.x.T, dout)

            dx = dx_p - dx_n

        else:
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)

        dx = dx.reshape(*self.original_x_shape)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    @numba.jit
    def forward(self, x, t):
        self.t = t
        # print("1ban")
        self.y = softmax(x)
        # print("2ban")
        self.loss = cross_entropy_error(self.y, self.t)
        # print("3ban")

        return self.loss

    @numba.jit
    def forward_acc(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def forward_sa(self, x, t, random_mask, sa):
        self.t = t
        self.y = softmax(x)
        #random_num = [0,1,2,3,4,5,6,7,8,9]
        #random_mask = random.choice(random_num,fr,replace=False)
        if sa == 0:
            for i in random_mask:
                self.y[i] = 0.000000000000000001
        elif sa == 1:
            for i in random_mask:
                self.y[i] = 1

        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def forward_sa_acc(self, x, t, random_mask, sa):

        self.t = t
        self.y = softmax(x)
        #random_num = [0,1,2,3,4,5,6,7,8,9]
        #random_mask = random.choice(random_num,fr,replace=False)
        if sa == 0:
            for i in random_mask:
                self.y[i] = 0.000000000000000001
        elif sa == 1:
            for i in random_mask:
                self.y[i] = 1

        return(self.y)

    @numba.jit
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            self.y = np.asarray(self.y)
            self.t = np.asarray(self.t)
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx = np.asarray(dx)
            self.t = np.asarray(self.t)
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def forward_acc(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

