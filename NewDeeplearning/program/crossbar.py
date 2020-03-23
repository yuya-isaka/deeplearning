
# coding: utf-8

import numpy as np
import copy

# X-ABFT，提案手法で行われるオンラインテストを実現するクラスファイル
# ニューラルネットワークの重み行列とテストブロックのサイズを与えると，テストを行ってくれる
class Crossbar(object):

    def __init__(self, W, row_size, column_size, row_test_size=2, column_test_size=2, fault_rate=0.0, prop=False, seed=0):
        self.row_size         = row_size
        self.column_size      = column_size
        self.row_test_size    = row_test_size
        self.column_test_size = column_test_size
        self.fault_rate       = fault_rate
        self.prop = prop
        self.tuuka = False

        self.seed = seed

        self.W = W

        self.column_weight = np.arange(1, self.column_test_size+1)
        self.row_weight    = np.arange(1, self.row_test_size+1)

        self.column_amari1 = False
        self.row_amari1    = False

        self.per_column = 0
        self.per_row = 0

        self.all_cell_num = self.W.shape[0] * self.W.shape[1]
        self.cell_fault_rate = 0

        if self.column_size % self.column_test_size == 1:
            self.column_amari1 = True
            self.per_column = int(np.floor(self.column_size / self.column_test_size))
            self.column_check_sum = np.zeros((self.row_size, int(np.floor(self.column_size/self.column_test_size)) * 2))
        else:
            self.per_column = int(np.ceil(self.column_size / self.column_test_size))
            self.column_check_sum = np.zeros((self.row_size, int(np.ceil(self.column_size/self.column_test_size)) * 2))

        if self.row_size % self.row_test_size == 1:
            self.row_amari1 = True
            self.per_row    = int(np.floor(self.row_size / self.row_test_size))
            self.row_check_sum    = np.zeros((int(np.floor(self.row_size/self.row_test_size)) * 2, self.column_size))
        else:
            self.per_row    = int(np.ceil(self.row_size / self.row_test_size))
            self.row_check_sum    = np.zeros((int(np.ceil(self.row_size/self.row_test_size)) * 2, self.column_size))

        self.fault_map     = np.full((self.row_size, self.column_size), -1)
        self.fault_value   = np.zeros((self.row_size, self.column_size))
        self.fault_locate  = np.zeros((self.row_size, self.column_size))

        self.comp_fault_map = np.full((self.row_size, self.column_size), 0)
        self.correction_map = np.zeros((self.row_size, self.column_size))

        self.fault_map_row_check_sum      = np.full((self.row_check_sum.shape[0], self.row_check_sum.shape[1]), -1)
        self.fault_value_row_check_sum    = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))
        self.fault_locate_row_check_sum   = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.comp_fault_map_row_check_sum = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))
        self.correction_map_row_check_sum = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.fault_map_column_check_sum      = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), -1)
        self.fault_value_column_check_sum    = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))
        self.fault_locate_column_check_sum   = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.comp_fault_map_column_check_sum = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), 0)
        self.correction_map_column_check_sum = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.W_copy = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.column_check_sum_copy = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))
        self.row_check_sum_copy    = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.new_fault_map                  = np.zeros((self.fault_map.shape[0], self.fault_map.shape[1]))
        self.new_fault_map_row_check_sum    = np.zeros((self.fault_map_row_check_sum.shape[0], self.fault_map_row_check_sum.shape[1]))
        self.new_fault_map_column_check_sum = np.zeros((self.fault_map_column_check_sum.shape[0], self.fault_map_column_check_sum.shape[1]))

        self.fault_rate = None

    def initialize(self):

        self.W = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.column_check_sum = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))
        self.row_check_sum    = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.fault_map     = np.full((self.row_size, self.column_size), -1)
        self.fault_value   = np.zeros((self.row_size, self.column_size))
        self.fault_locate  = np.zeros((self.row_size, self.column_size))

        self.comp_fault_map = np.full((self.row_size, self.column_size), 0)
        self.correction_map = np.zeros((self.row_size, self.column_size))

        self.fault_map_row_check_sum      = np.full((self.row_check_sum.shape[0], self.row_check_sum.shape[1]), -1)
        self.fault_value_row_check_sum    = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))
        self.fault_locate_row_check_sum   = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.comp_fault_map_row_check_sum = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))
        self.correction_map_row_check_sum = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.fault_map_column_check_sum      = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), -1)
        self.fault_value_column_check_sum    = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))
        self.fault_locate_column_check_sum   = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.comp_fault_map_column_check_sum = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), 0)
        self.correction_map_column_check_sum = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.W_copy = np.zeros((self.W.shape[0], self.W.shape[1]))
        self.column_check_sum_copy = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))
        self.row_check_sum_copy    = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.new_fault_map                  = np.zeros((self.fault_map.shape[0], self.fault_map.shape[1]))
        self.new_fault_map_row_check_sum    = np.zeros((self.fault_map_row_check_sum.shape[0], self.fault_map_row_check_sum.shape[1]))
        self.new_fault_map_column_check_sum = np.zeros((self.fault_map_column_check_sum.shape[0], self.fault_map_column_check_sum.shape[1]))

    def set_check_sum(self, W):

        self.W = W

        for i in range(self.per_column):
            if self.column_amari1 == False:
                if (self.column_size % self.column_test_size != 0) and (i == int(np.ceil(self.column_size/self.column_test_size))-1):
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):int((i+1)*self.column_test_size)], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)]
                    tmp = np.dot(tmp, self.column_weight[:int(self.column_size % self.column_test_size)])
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)
                else:
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):int((i+1)*self.column_test_size)], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)]
                    tmp = np.dot(tmp, self.column_weight)
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)

            else:
                if (i == self.per_column - 1):
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):]
                    tmp = np.dot(tmp, np.arange(1, self.column_test_size+2))
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)
                else:
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):int((i+1)*self.column_test_size)], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)]
                    tmp = np.dot(tmp, self.column_weight)
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)

        for j in range(self.per_row):
            if self.row_amari1 == False:
                if (self.row_size % self.row_test_size != 0) and (j == int(np.ceil(self.row_size/self.row_test_size))-1):
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :]
                    tmp = np.dot(self.row_weight[:int(self.row_size % self.row_test_size)], tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)
                else:
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :]
                    tmp = np.dot(self.row_weight, tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)

            else:
                if (j == self.per_row - 1):
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):, :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):, :]
                    # print("tmp is : "+str(tmp))
                    tmp = np.dot(np.arange(1, self.row_test_size+2), tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)
                else:
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :]
                    tmp = np.dot(self.row_weight, tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)

    # 浮動小数点エラー回避．
    def set_check_sum2(self, W):

        self.W = W
        # print("self.W is : "+str(self.W))

        for i in range(self.per_column):
            if self.column_amari1 == False:
                if (self.column_size % self.column_test_size != 0) and (i == int(np.ceil(self.column_size/self.column_test_size))-1):
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):int((i+1)*self.column_test_size)], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)]
                    tmp = np.dot(tmp, self.column_weight[:int(self.column_size % self.column_test_size)])
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)
                else:
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):int((i+1)*self.column_test_size)], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)]
                    tmp = np.dot(tmp, self.column_weight)
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)

            else:
                if (i == self.per_column - 1):
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):]
                    tmp = np.dot(tmp, np.arange(1, self.column_test_size+2))
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)
                else:
                    self.column_check_sum[:, int(i*2)] = np.round(np.sum(self.W[:, int(i*self.column_test_size):int((i+1)*self.column_test_size)], axis=1), decimals=6)
                    tmp = self.W[:,int(i*self.column_test_size):int((i+1)*self.column_test_size)]
                    tmp = np.dot(tmp, self.column_weight)
                    self.column_check_sum[:, int(i*2+1)] += np.round(tmp, decimals=6)

        for j in range(self.per_row):
            if self.row_amari1 == False:
                if (self.row_size % self.row_test_size != 0) and (j == int(np.ceil(self.row_size/self.row_test_size))-1):
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :]
                    tmp = np.dot(self.row_weight[:int(self.row_size % self.row_test_size)], tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)
                else:
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :]
                    tmp = np.dot(self.row_weight, tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)

            else:
                if (j == self.per_row - 1):
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):, :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):, :]
                    # print("tmp is : "+str(tmp))
                    tmp = np.dot(np.arange(1, self.row_test_size+2), tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)
                else:
                    self.row_check_sum[int(j*2), :] = np.round(np.sum(self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :], axis=0), decimals=6)
                    tmp = self.W[int(j*self.row_test_size):int((j+1)*self.row_test_size), :]
                    tmp = np.dot(self.row_weight, tmp)
                    self.row_check_sum[int(j*2+1), :] += np.round(tmp, decimals=6)

    # 浮動小数点エラー回避
    def set_fault_map(self, fault_value, fault_locate, row_fault_value, row_fault_locate, column_fault_value, column_fault_locate):

        self.fault_map     = np.full((self.row_size, self.column_size), -1)
        self.fault_value   = np.ceil(fault_value*1000)
        self.fault_locate  = fault_locate

        self.fault_map      = np.where(self.fault_locate == 1, self.fault_value, self.fault_map)

        self.comp_fault_map = np.full((self.row_size, self.column_size), 0)
        self.correction_map = np.zeros((self.row_size, self.column_size))

        self.fault_map_row_check_sum      = np.full((self.row_check_sum.shape[0], self.row_check_sum.shape[1]), -1)
        self.fault_value_row_check_sum    = np.ceil(row_fault_value*1000)
        self.fault_locate_row_check_sum   = row_fault_locate

        self.fault_map_row_check_sum      = np.where(self.fault_locate_row_check_sum == 1, self.fault_value_row_check_sum, self.fault_map_row_check_sum)

        self.comp_fault_map_row_check_sum = np.full((self.row_check_sum.shape[0], self.row_check_sum.shape[1]), 0)
        self.correction_map_row_check_sum = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.fault_map_column_check_sum      = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), -1)
        self.fault_value_column_check_sum    = np.ceil(column_fault_value*1000)
        self.fault_locate_column_check_sum   = column_fault_locate

        self.fault_map_column_check_sum      = np.where(self.fault_locate_column_check_sum == 1, self.fault_value_column_check_sum, self.fault_map_column_check_sum)

        self.comp_fault_map_column_check_sum = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), 0)
        self.correction_map_column_check_sum = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.W_copy = copy.deepcopy(self.W)
        self.column_check_sum_copy = copy.deepcopy(self.column_check_sum)
        self.row_check_sum_copy = copy.deepcopy(self.row_check_sum)

        self.new_fault_map                  = np.zeros((self.fault_map.shape[0], self.fault_map.shape[1]))
        self.new_fault_map_row_check_sum    = np.zeros((self.fault_map_row_check_sum.shape[0], self.fault_map_row_check_sum.shape[1]))
        self.new_fault_map_column_check_sum = np.zeros((self.fault_map_column_check_sum.shape[0], self.fault_map_column_check_sum.shape[1]))

        # self.fault_rate = np.sum(self.fault_locate) / self.all_cell_num

    # 浮動小数点エラー回避
    def set_fault_map2(self, fault_value, fault_locate, row_fault_value, row_fault_locate, column_fault_value, column_fault_locate):

        self.seed += 1
        np.random.seed(self.seed)
        self.set_check_sum2(np.random.randint(0,101,(int(self.W.shape[0]),int(self.W.shape[1]))))

        self.fault_map     = np.full((self.row_size, self.column_size), -1)
        self.seed += 1
        np.random.seed(self.seed)
        self.fault_value   = np.random.randint(0,101,(int(self.W.shape[0]),int(self.W.shape[1])))
        self.fault_locate  = fault_locate

        self.fault_map      = np.where(self.fault_locate == 1, self.fault_value, self.fault_map)

        self.comp_fault_map = np.full((self.row_size, self.column_size), 0)
        self.correction_map = np.zeros((self.row_size, self.column_size))

        self.fault_map_row_check_sum      = np.full((self.row_check_sum.shape[0], self.row_check_sum.shape[1]), -1)
        self.seed += 1
        np.random.seed(self.seed)
        self.fault_value_row_check_sum    = np.random.randint(0,101,(int(self.row_check_sum.shape[0]),int(self.row_check_sum.shape[1])))
        self.fault_locate_row_check_sum   = row_fault_locate

        self.fault_map_row_check_sum      = np.where(self.fault_locate_row_check_sum == 1, self.fault_value_row_check_sum, self.fault_map_row_check_sum)

        self.comp_fault_map_row_check_sum = np.full((self.row_check_sum.shape[0], self.row_check_sum.shape[1]), 0)
        self.correction_map_row_check_sum = np.zeros((self.row_check_sum.shape[0], self.row_check_sum.shape[1]))

        self.fault_map_column_check_sum      = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), -1)
        self.seed += 1
        np.random.seed(self.seed)
        self.fault_value_column_check_sum    = np.random.randint(0,101,(int(self.column_check_sum.shape[0]),int(self.column_check_sum.shape[1])))
        self.fault_locate_column_check_sum   = column_fault_locate

        self.fault_map_column_check_sum      = np.where(self.fault_locate_column_check_sum == 1, self.fault_value_column_check_sum, self.fault_map_column_check_sum)

        self.comp_fault_map_column_check_sum = np.full((self.column_check_sum.shape[0], self.column_check_sum.shape[1]), 0)
        self.correction_map_column_check_sum = np.zeros((self.column_check_sum.shape[0], self.column_check_sum.shape[1]))

        self.W_copy = copy.deepcopy(self.W)
        self.column_check_sum_copy = copy.deepcopy(self.column_check_sum)
        self.row_check_sum_copy = copy.deepcopy(self.row_check_sum)

        self.new_fault_map                  = np.zeros((self.fault_map.shape[0], self.fault_map.shape[1]))
        self.new_fault_map_row_check_sum    = np.zeros((self.fault_map_row_check_sum.shape[0], self.fault_map_row_check_sum.shape[1]))
        self.new_fault_map_column_check_sum = np.zeros((self.fault_map_column_check_sum.shape[0], self.fault_map_column_check_sum.shape[1]))

    def set_W(self, W):
        self.W = W

    def fault_injection(self):
        self.W = np.where(self.fault_map != -1, self.fault_map, self.W)
        self.column_check_sum = np.where(self.fault_map_column_check_sum != -1, self.fault_map_column_check_sum, self.column_check_sum)
        self.row_check_sum    = np.where(self.fault_map_row_check_sum != -1, self.fault_map_row_check_sum, self.row_check_sum)

        self.W_copy = self.W_copy - self.W
        self.column_check_sum_copy = self.column_check_sum_copy - self.column_check_sum
        self.row_check_sum_copy = self.row_check_sum_copy - self.row_check_sum

    def correction(self):
        self.W = self.W + self.correction_map

    def test(self):
        # $B%F%9%H%Y%/%H%k@8@.(B
        test_vector = np.array([np.array([pow(2, i*0) for i in range(self.row_test_size)]),
                                np.array([pow(2, i*1) for i in range(self.row_test_size)]),
                                np.array([pow(2, i*2) for i in range(self.row_test_size)]),
                                np.array([pow(2, i*3) for i in range(self.row_test_size)])])

        test_vector_amari1 = np.array([np.array([pow(2, i*0) for i in range(self.row_test_size+1)]),
                                       np.array([pow(2, i*1) for i in range(self.row_test_size+1)]),
                                       np.array([pow(2, i*2) for i in range(self.row_test_size+1)]),
                                       np.array([pow(2, i*3) for i in range(self.row_test_size+1)])])

        # $BJ,3d$7$?J,$@$1!$%F%9%H$r7+$jJV$9(B
        # $BNsJ,3d9MN8$9$k$+$I$&$+(B
        for i in range(self.per_row):
            for j in range(self.per_column):
                if self.column_amari1 == False and self.row_amari1 == False:
                    if (self.row_size % self.row_test_size != 0) and (i == self.per_row-1):
                        O1     = np.dot(test_vector[0,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O1_sum = np.dot(test_vector[0,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O2     = np.dot(test_vector[1,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O2_sum = np.dot(test_vector[1,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O3     = np.dot(test_vector[2,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O3_sum = np.dot(test_vector[2,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O4     = np.dot(test_vector[3,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O4_sum = np.dot(test_vector[3,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                    else:
                        O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O1_sum = np.dot(test_vector[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O2_sum = np.dot(test_vector[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O3_sum = np.dot(test_vector[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O4_sum = np.dot(test_vector[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                elif self.column_amari1 == False and self.row_amari1 == True:
                    if (i == self.per_row-1):
                        O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O1_sum = np.dot(test_vector_amari1[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                        O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O2_sum = np.dot(test_vector_amari1[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                        O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O3_sum = np.dot(test_vector_amari1[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                        O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O4_sum = np.dot(test_vector_amari1[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])

                    else:
                        O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O1_sum = np.dot(test_vector[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O2_sum = np.dot(test_vector[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O3_sum = np.dot(test_vector[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                        O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        O4_sum = np.dot(test_vector[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                elif self.column_amari1 == True and self.row_amari1 == False:
                    if (self.row_size % self.row_test_size != 0) and (i == self.per_row-1):
                        if (j == self.per_column-1):
                            O1     = np.dot(test_vector[0,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O1_sum = np.dot(test_vector[0,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector[1,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O2_sum = np.dot(test_vector[1,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector[2,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O3_sum = np.dot(test_vector[2,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector[3,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O4_sum = np.dot(test_vector[3,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                        else:
                            O1     = np.dot(test_vector[0,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O1_sum = np.dot(test_vector[0,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector[1,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O2_sum = np.dot(test_vector[1,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector[2,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O3_sum = np.dot(test_vector[2,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector[3,:int(self.row_size % self.row_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O4_sum = np.dot(test_vector[3,:int(self.row_size % self.row_test_size)], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                    else:
                        if (j == self.per_column-1):
                            O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O1_sum = np.dot(test_vector[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O2_sum = np.dot(test_vector[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O3_sum = np.dot(test_vector[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O4_sum = np.dot(test_vector[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                        else:
                            O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O1_sum = np.dot(test_vector[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O2_sum = np.dot(test_vector[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O3_sum = np.dot(test_vector[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O4_sum = np.dot(test_vector[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                elif self.column_amari1 == True and self.row_amari1 == True:
                    if (i == self.per_row-1):
                        if (j == self.per_column-1):
                            O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):])
                            O1_sum = np.dot(test_vector_amari1[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):])
                            O2_sum = np.dot(test_vector_amari1[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):])
                            O3_sum = np.dot(test_vector_amari1[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):])
                            O4_sum = np.dot(test_vector_amari1[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])

                        else:
                            O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O1_sum = np.dot(test_vector_amari1[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O2_sum = np.dot(test_vector_amari1[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O3_sum = np.dot(test_vector_amari1[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O4_sum = np.dot(test_vector_amari1[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*2):int((j*2)+2)])

                    else:
                        if (j == self.per_column-1):
                            O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O1_sum = np.dot(test_vector[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O2_sum = np.dot(test_vector[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O3_sum = np.dot(test_vector[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):])
                            O4_sum = np.dot(test_vector[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                        else:
                            O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O1_sum = np.dot(test_vector[0,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O2_sum = np.dot(test_vector[1,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O3_sum = np.dot(test_vector[2,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])
                            O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                            O4_sum = np.dot(test_vector[3,:], self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)])

                # print("O1 is : "+str(O1))
                # print("O2 is : "+str(O2))
                # print("O3 is : "+str(O3))
                # print("O4 is : "+str(O4))

                # print("O1_sum is : "+str(O1_sum))
                # print("O2_sum is : "+str(O2_sum))
                # print("O3_sum is : "+str(O3_sum))
                # print("O4_sum is : "+str(O4_sum))

                x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num = self.signature_analysis(O1, O2, O3, O4, O1_sum, O2_sum, O3_sum, O4_sum, i, j)

                # print("test_vector is : \n"+str(test_vector))
                # print("self.W is : \n"+str(self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)]))
                # print("self.column_check_sum is : \n"+str(self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)]))

                #print("test_vector[0,:] is : \n"+str(test_vector[0,:]))
                #print("test_vector[1,:] is : \n"+str(test_vector[1,:]))
                #print("test_vector[2,:] is : \n"+str(test_vector[2,:]))
                #print("test_vector[3,:] is : \n"+str(test_vector[3,:]))

                #print("O1 is : "+str(O1))
                #print("O2 is : "+str(O2))
                #print("O3 is : "+str(O3))
                #print("O4 is : "+str(O4))

                #print("O1_sum is : "+str(O1_sum))
                #print("O2_sum is : "+str(O2_sum))
                #print("O3_sum is : "+str(O3_sum))
                #print("O4_sum is : "+str(O4_sum))

                #if (omomi_fault_num >= 1):
                #    print("x1 is : "+str(x1))
                #    print("y1 is : "+str(y1))
                #    print("d1 is : "+str(d1))
                #    if (omomi_fault_num >= 2):
                #        print("x2 is : "+str(x2))
                #        print("y2 is : "+str(y2))
                #        print("d2 is : "+str(d2))

                #if (col_checksum_fnum >= 1):
                #    print("x1_col is : "+str(x1_col))
                #    print("y1_col is : "+str(y1_col))
                #    print("d1_col is : "+str(d1_col))

                #    if (col_checksum_fnum >= 2):
                #        print("x2_col is : "+str(x2_col))
                #        print("y2_col is : "+str(y2_col))
                #        print("d2_col is : "+str(d2_col))

                #if (row_checksum_fnum >= 1):
                #    print("x1_row is : "+str(x1_row))
                #    print("y1_row is : "+str(y1_row))
                #    print("d1_row is : "+str(d1_row))
                #    if (row_checksum_fnum >= 2):
                #        print("x2_row is : "+str(x2_row))
                #        print("y2_row is : "+str(y2_row))
                #        print("d2_row is : "+str(d2_row))

                if (possible != False) and (fault_free != True):
                    if (omomi_fault_num >= 1 and (self.comp_fault_map.shape[0] > int(i*self.row_test_size+x1)) and (self.comp_fault_map.shape[1] > int(j*self.column_test_size+y1))):
                        self.comp_fault_map[int(i*self.row_test_size+x1), int(j*self.column_test_size+y1)] = 1
                        self.correction_map[int(i*self.row_test_size+x1), int(j*self.column_test_size+y1)] = d1

                        if (omomi_fault_num == 2 and (self.comp_fault_map.shape[0] > int(i*self.row_test_size+x2)) and (self.comp_fault_map.shape[1] > int(j*self.column_test_size+y2))):
                            self.comp_fault_map[int(i*self.row_test_size+x2), int(j*self.column_test_size+y2)] = 1
                            self.correction_map[int(i*self.row_test_size+x2), int(j*self.column_test_size+y2)] = d2

                    if (col_checksum_fnum >= 1 and (self.comp_fault_map_column_check_sum.shape[0] > int(i*self.row_test_size+x1_col)) and (self.comp_fault_map_column_check_sum.shape[1] > int(j*2+y1_col))):
                        self.comp_fault_map_column_check_sum[int(i*self.row_test_size+x1_col), int(j*2+y1_col)] = 1
                        self.correction_map_column_check_sum[int(i*self.row_test_size+x1_col), int(j*2+y1_col)] = d1_col

                        if (col_checksum_fnum == 2 and (self.comp_fault_map_column_check_sum.shape[0] > int(i*self.row_test_size+x2_col)) and (self.comp_fault_map_column_check_sum.shape[1] > int(j*2+y2_col))):
                            self.comp_fault_map_column_check_sum[int(i*self.row_test_size+x2_col), int(j*2+y2_col)] = 1
                            self.correction_map_column_check_sum[int(i*self.row_test_size+x2_col), int(j*2+y2_col)] = d2_col

                    if (row_checksum_fnum >= 1 and (self.comp_fault_map_row_check_sum.shape[0] > int(i*2+x1_row)) and (self.comp_fault_map_row_check_sum.shape[1] > int(j*self.column_test_size+y1_row))):
                        self.comp_fault_map_row_check_sum[int(i*2+x1_row), int(j*self.column_test_size+y1_row)] = 1
                        self.correction_map_row_check_sum[int(i*2+x1_row), int(j*self.column_test_size+y1_row)] = d1_row

                        if (row_checksum_fnum == 2 and (self.comp_fault_map_row_check_sum.shape[0] > int(i*2+x2_row)) and (self.comp_fault_map_row_check_sum.shape[1] > int(j*self.column_test_size+y2_row))):
                            self.comp_fault_map_row_check_sum[int(i*2+x2_row), int(j*self.column_test_size+y2_row)] = 1
                            self.correction_map_row_check_sum[int(i*2+x2_row), int(j*self.column_test_size+y2_row)] = d2_row

                    #if (self.tuuka == False and (np.sum(self.fault_locate_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)]) + np.sum(self.fault_locate[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)]) <= 2)):
                        # $B8e$G$3$NF0:n$r:n$k(B
                        # self.only_column_test(i, j)
                        # row_checksum_fnum = np.sum(self.fault_locate_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                        #print("row_checksum_fnum : "+str(row_checksum_fnum))
                        #if (row_checksum_fnum >= 1 and (self.comp_fault_map_row_check_sum.shape[0] > int(i*2+y1_row)) and (self.comp_fault_map_row_check_sum.shape[1] > int(j*self.column_test_size+x1_row))):
                        #    self.comp_fault_map_row_check_sum[int(i*2+y1_row), int(j*self.column_test_size+x1_row)] = 1
                        #    if (row_checksum_fnum == 2 and (self.comp_fault_map_row_check_sum.shape[0] > int(i*2+y2_row)) and (self.comp_fault_map_row_check_sum.shape[1] > int(j*self.column_test_size+x2_row))):
                        #        self.comp_fault_map_row_check_sum[int(i*2+y2_row), int(j*self.column_test_size+x2_row)] = 1
                        #if (row_checksum_fnum <= 2):
                            #self.comp_fault_map_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)] = self.fault_locate_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)]

                #if (self.prop == True) and (possible != False) and (fault_free == True) and (self.tuuka == False):
                    # $B8e$G$3$NF0:n$r:n$k(B
                    # self.only_column_test(i, j)
                    #row_checksum_fnum = np.sum(self.fault_locate_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)])
                    #print("row_checksum_fnum : "+str(row_checksum_fnum))
                    #if (row_checksum_fnum >= 1 and (self.comp_fault_map_row_check_sum.shape[0] > int(i*2+y1_row)) and (self.comp_fault_map_row_check_sum.shape[1] > int(j*self.column_test_size+x1_row))):
                    #    self.comp_fault_map_row_check_sum[int(i*2+y1_row), int(j*self.column_test_size+x1_row)] = 1
                    #    if (row_checksum_fnum == 2 and (self.comp_fault_map_row_check_sum.shape[0] > int(i*2+y2_row)) and (self.comp_fault_map_row_check_sum.shape[1] > int(j*self.column_test_size+x2_row))):
                    #        self.comp_fault_map_row_check_sum[int(i*2+y2_row), int(j*self.column_test_size+x2_row)] = 1
                    #if (row_checksum_fnum <= 2):
                    #    self.comp_fault_map_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)] = self.fault_locate_row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)]

                    #self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

                #if self.yoko == True:

                #print("self.fault_locate is : \n"+str(self.fault_locate))
                #print("self.fault_locate_column_check_sum is : \n"+str(self.fault_locate_column_check_sum))
                #print("self.fault_locate_row_check_sum.T is : \n"+str(self.fault_locate_row_check_sum.T))

                #print("self.comp_fault_map is : \n"+str(self.comp_fault_map))
                #print("self.comp_fault_map_column_check_sum is : \n"+str(self.comp_fault_map_column_check_sum))
                #print("self.comp_fault_map_row_check_sum.T is : \n"+str(self.comp_fault_map_row_check_sum.T))

                #print("self.correction_map is : \n"+str(self.correction_map))
                #print("self.correction_map_column_check_sum is : \n"+str(self.correction_map_column_check_sum))
                #print("self.correction_map_row_check_sum.T is : \n"+str(self.correction_map_row_check_sum.T))

                #print("self.W_copy is : \n"+str(self.W_copy))
                #print("self.column_check_sum_copy is : \n"+str(self.column_check_sum_copy))
                #print("self.row_check_sum_copy is : \n"+str(self.row_check_sum_copy.T))

        if (self.prop == False):

            exist_fault               = np.sum(self.fault_locate) + np.sum(self.fault_locate_column_check_sum)
            collectly_localized_fault = np.sum(np.logical_and(self.fault_locate, self.comp_fault_map)) + np.sum(np.logical_and(self.fault_locate_column_check_sum, self.comp_fault_map_column_check_sum))

            localized_fault = np.sum(self.comp_fault_map) + np.sum(self.comp_fault_map_column_check_sum)

            self.new_fault_map                  = np.where((self.comp_fault_map == 0) & (self.fault_locate == 1), 1, 0)
            self.new_fault_map_row_check_sum    = np.where((self.comp_fault_map_row_check_sum == 0) & (self.fault_locate_row_check_sum == 1), 1, 0)
            self.new_fault_map_column_check_sum = np.where((self.comp_fault_map_column_check_sum == 0) & (self.fault_locate_column_check_sum == 1), 1, 0)

            #print("collectly_localized_fault is : \n"+str(collectly_localized_fault))

            recall    = collectly_localized_fault / exist_fault
            precision = collectly_localized_fault / localized_fault

            only_exist_fault               = np.sum(self.fault_locate) + np.sum(self.fault_locate_column_check_sum)
            only_collectly_localized_fault = np.sum(np.logical_and(self.fault_locate, self.comp_fault_map)) + np.sum(np.logical_and(self.fault_locate_column_check_sum, self.comp_fault_map_column_check_sum))

            only_localized_fault = np.sum(self.comp_fault_map) + np.sum(self.comp_fault_map_column_check_sum)

            only_recall = only_collectly_localized_fault / only_exist_fault
            only_precision = only_collectly_localized_fault / only_localized_fault

            #print("self.fault_locate is : \n"+str(self.fault_locate))
            #print("self.fault_locate_column_check_sum is : \n"+str(self.fault_locate_column_check_sum))
            #print("self.fault_locate_row_check_sum is : \n"+str(self.fault_locate_row_check_sum.T))
            #print("self.comp_fault_map is : \n"+str(self.comp_fault_map))
            #print("self.comp_fault_map_column_check_sum is : \n"+str(self.comp_fault_map_column_check_sum))
            #print("self.comp_fault_map_row_check_sum is : \n"+str(self.comp_fault_map_row_check_sum.T))

        else:
            exist_fault               = np.sum(self.fault_locate) + np.sum(self.fault_locate_column_check_sum) + np.sum(self.fault_locate_row_check_sum)
            # self.comp_fault_map が診断した故障位置（らしい）
            collectly_localized_fault = np.sum(np.logical_and(self.fault_locate, self.comp_fault_map)) + np.sum(np.logical_and(self.fault_locate_column_check_sum, self.comp_fault_map_column_check_sum)) + np.sum(np.logical_and(self.fault_locate_row_check_sum, self.comp_fault_map_row_check_sum))
            localized_fault = np.sum(self.comp_fault_map) + np.sum(self.comp_fault_map_column_check_sum) + np.sum(self.comp_fault_map_row_check_sum)

            only_exist_fault               = np.sum(self.fault_locate) + np.sum(self.fault_locate_column_check_sum)
            only_collectly_localized_fault = np.sum(np.logical_and(self.fault_locate, self.comp_fault_map)) + np.sum(np.logical_and(self.fault_locate_column_check_sum, self.comp_fault_map_column_check_sum))

            only_localized_fault = np.sum(self.comp_fault_map) + np.sum(self.comp_fault_map_column_check_sum)

            #print("collectly_localized_fault is : \n"+str(collectly_localized_fault))

            self.new_fault_map                  = np.where((self.comp_fault_map == 0) & (self.fault_locate == 1), 1, 0)
            self.new_fault_map_row_check_sum    = np.where((self.comp_fault_map_row_check_sum == 0) & (self.fault_locate_row_check_sum == 1), 1, 0)
            self.new_fault_map_column_check_sum = np.where((self.comp_fault_map_column_check_sum == 0) & (self.fault_locate_column_check_sum == 1), 1, 0)

            recall    = collectly_localized_fault / exist_fault
            precision = collectly_localized_fault / localized_fault

            only_recall = only_collectly_localized_fault / only_exist_fault
            only_precision = only_collectly_localized_fault / only_localized_fault

        if (np.isnan(recall)):
            recall = 1
        if (np.isnan(precision)):
            precision = 1

        print("recall is : "+str(recall))
        print("precision is : "+str(precision))

        self.cell_fault_rate = np.sum(self.new_fault_map) / self.all_cell_num
        print("fault_cell_num (before test) is : "+str(np.sum(self.fault_locate)))
        print("fault_cell_num (after test) is : "+str(np.sum(self.new_fault_map)))

        # print("only_recall is : "+str(only_recall))
        # print("only_precision is : "+str(only_precision))

        return recall, precision

        # print("self.comp_fault_map is : \n"+str(self.comp_fault_map))
        # print("self.comp_fault_map_column_check_sum is : \n"+str(self.comp_fault_map_column_check_sum))
        # print("self.comp_fault_map_row_check_sum is : \n"+str(self.comp_fault_map_row_check_sum.T))

    def signature_analysis(self, O1, O2, O3, O4, O1_sum, O2_sum, O3_sum, O4_sum, i, j):

        self.tuuka = False

        row_check      = False
        row_only_check = False
        #possible   = True
        #fault_free = False
        row_checksum_fnum = 0
        col_checksum_fnum = 0
        self.yoko = False

        omomi_1    = False

        check      = False
        only_check = False
        possible   = True
        fault_free = False
        omomi_fault_num  = 0

        # tuika for row amari
        tfra = 0
        # tuika for column amari
        tfca = 0

        if (self.row_amari1 == True and i == self.per_row-1):
            tfra = 1

        if (self.column_amari1 == True and j == self.per_column-1):
            tfca = 1

        #print("O1 is : "+str(O1))
        #print("O2 is : "+str(O2))
        #print("O1_sum is : "+str(O1_sum))
        #print("O2_sum is : "+str(O2_sum))

        WG = np.arange(1, O1.shape[0]+1)

        A1 = np.round(np.sum(O1) - O1_sum[0], decimals=6)
        A2 = np.round(np.sum(O2) - O2_sum[0], decimals=6)
        A3 = np.round(np.sum(O3) - O3_sum[0], decimals=6)
        A4 = np.round(np.sum(O4) - O4_sum[0], decimals=6)

        B1 = np.round(np.sum(np.dot(WG, O1)) - O1_sum[1], decimals=6)
        B2 = np.round(np.sum(np.dot(WG, O2)) - O2_sum[1], decimals=6)
        B3 = np.round(np.sum(np.dot(WG, O3)) - O3_sum[1], decimals=6)
        B4 = np.round(np.sum(np.dot(WG, O4)) - O4_sum[1], decimals=6)

        # print("A1 is : "+str(A1))
        # print("A2 is : "+str(A2))
        # print("A3 is : "+str(A3))
        # print("A4 is : "+str(A4))

        # print("B1 is : "+str(B1))
        # print("B2 is : "+str(B2))
        # print("B3 is : "+str(B3))
        # print("B4 is : "+str(B4))

        # print("O1 is : "+str(O1))
        # print("O2 is : "+str(O2))
        # print("O3 is : "+str(O3))
        # print("O4 is : "+str(O4))

        # print("O1_sum is : "+str(O1_sum))
        # print("O2_sum is : "+str(O2_sum))
        # print("O3_sum is : "+str(O3_sum))
        # print("O4_sum is : "+str(O4_sum))

        #print("fault_locate is : "+str(self.fault_locate[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)]))
        #print("fault_locate_column_check_sum is : "+str(self.fault_locate_column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)]))
        eta = 0
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        d1 = 0
        d2 = 0

        x1_col = 0
        x2_col = 0
        y1_col = 0
        y2_col = 0
        d1_col = 0
        d2_col = 0

        x1_row = 0
        x2_row = 0
        y1_row = 0
        y2_row = 0
        d1_row = 0
        d2_row = 0

        if int(A1 == A2 == A3 == A4 == 0) and int(B1 == B2 == B3 == B4 == 0):
            #print("fault_free")
            fault_free = True

            if (self.prop == True) and (possible == True):
                x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

            if (row_checksum_fnum >= 1):
                fault_free = False
            return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

        if (int(A1 == A2 == A3 == A4 == 0) or int(B1 == B2 == B3 == B4 == 0)):
            if(int(A1 == A2 == A3 == A4 == 0)):
                only_check = True
                check = True
                if (B1*B4 - B2*B3 == 0) and (B1*B3 - B2*B2 == 0):
                    # print("kensa 1 a")
                    omomi_fault_num = 0
                    col_checksum_fnum = 1
                    y1_col = 1
                    # if (B1 != 0):
                    x1_col = int(np.log2(int(B2/B1)) + 1)
                    # else:
                    #x1_col = 0
                    d1_col = B1

                    # $BD"?,$r9g$o$;$k(B
                    x1_col -= 1

                    if x1 >= self.row_test_size or x1_col >= self.row_test_size or y1 >= self.column_test_size or y1_col >= self.column_test_size or np.isnan(x1) or np.isnan(x1_col) or np.isnan(y1) or np.isnan(y1_col):
                        possible = False
                    if x1 < 0 or x1_col < 0 or y1 < 0 or y1_col < 0:
                        possible = False

                    if (self.prop == True) and (possible == True):
                        x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

                else:
                    # print("kensa 2 a")
                    # possible = False
                    omomi_fault_num = 0
                    col_checksum_fnum = 2
                    y1_col = 1
                    y2_col = 1

                    if (int(B1*B3 - B2*B2)) != 0:
                        eta = int((B1*B4 - B2*B3)) / int((B1*B3 - B2*B2))
                        #print("eta is : "+str(eta))
                        #print("not np.isnan(eta) is : "+str(not np.isnan(eta)))

                        if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):

                            x1_col = int(np.floor(np.log2(eta)) + 1)
                            if ((eta - np.power(2, x1_col-1)) != 0):
                                # $B9TNs@8@.;~$K(Bcheck sum$BJ,$1$F$k$+$i!$$"$H$GD{@5$9$kI,MW$"$j(B($B7W;;MQ$K0l;~E*$KBeF~$7$F7W;;$9$k(B)$B$H!$;W$C$?$,!$(Bx2$B!J9T$N0LCV!K$J$i!$FC$KLdBj$J$+$C$?(B
                                #print("eta is :"+str(eta))
                                #print("x1_col is :"+str(x1_col))
                                #print("np.power(2, x1-col-1) is : "+str(np.power(2, x1_col-1)))
                                x2_col = int(np.log2(eta - np.power(2, x1_col-1))) + 1

                                d1_col = (-B2 + int(np.power(2.0, x2_col-1)*B1)) / (int(np.power(2.0, x2_col-1) - np.power(2.0, x1_col-1)))
                                d2_col = B1 - d1_col

                        # both = True
                        # $BD"?,$r9g$o$;$k(B
                    x1_col -= 1
                    x2_col -= 1

                    if x1_col >= self.row_test_size or x2_col >= self.row_test_size or y1_col >= self.column_test_size or y2_col >= self.column_test_size or np.isnan(x1_col) or np.isnan(x2_col) or np.isnan(y1_col) or np.isnan(y2_col):
                        possible = False
                    if x1_col < 0 or x2_col < 0 or y1_col < 0 or y2_col < 0:
                        possible = False

                    if (self.prop == True) and (possible == True):
                        x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

            elif(int(B1 == B2 == B3 == B4 == 0)):
                only_check = True
                check = True
                if (int(A1*A4 - A2*A3) == 0) and (int(A1*A3 - A2*A2) == 0):
                    # print("kensa 1 b")
                    omomi_fault_num = 0
                    col_checksum_fnum = 1
                    y1_col = 0
                    # if (A1 != 0):
                    x1_col = np.log2(int(A2/A1)) + 1
                    #else:
                    #    x1_col = 0
                    d1_col = A1

                    # $BD"?,$r9g$o$;$k(B
                    x1_col -= 1

                    both = True

                    if x1_col >= self.row_test_size or x2_col >= self.row_test_size or y1_col >= self.column_test_size or y2_col >= self.column_test_size or np.isnan(x1_col) or np.isnan(x2_col) or np.isnan(y1_col) or np.isnan(y2_col):
                        possible = False
                    if x1_col < 0 or x2_col < 0 or y1_col < 0 or y2_col < 0:
                        possible = False

                    if (self.prop == True) and (possible == True):
                        x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

                else:
                    #print("kensa 2 b")
                    # possible = False
                    omomi_fault_num = 0
                    col_checksum_fnum = 2
                    y1_col = 0
                    y2_col = 0

                    if (int(A1*A3 - A2*A2)) != 0:
                        eta = int((A1*A4 - A2*A3)) / int((A1*A3 - A2*A2))
                        #print("eta is : "+str(eta))
                        #print("not np.isnan(eta) is : "+str(not np.isnan(eta)))
                    
                        if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                            x1_col = int(np.floor(np.log2(eta)) + 1)
                            if ((eta - np.power(2, x1_col-1)) != 0):
                                # $B9TNs@8@.;~$K(Bcheck sum$BJ,$1$F$k$+$i!$$"$H$GD{@5$9$kI,MW$"$j(B($B7W;;MQ$K0l;~E*$KBeF~$7$F7W;;$9$k(B)$B$H!$;W$C$?$,!$(Bx2$B!J9T$N0LCV!K$J$i!$FC$KLdBj$J$+$C$?(B
                                #print("eta is :"+str(eta))
                                #print("x1_col is :"+str(x1_col))
                                #print("np.power(2, x1-col-1) is : "+str(np.power(2, x1_col-1)))
                                x2_col = int(np.log2(eta - np.power(2, x1_col-1))) + 1
                                #print("x2 is : "+str(x2))
                                d1_col = (-A2 + int(np.power(2.0, x2_col-1)*A1)) / (int(np.power(2.0, x2_col-1) - np.power(2.0, x1_col-1)))
                                d2_col = A1 - d1_col
                        
                    #both = True
                    # $BD"?,$r9g$o$;$k(B
                    x1_col -= 1
                    x2_col -= 1

                    if x1_col >= self.row_test_size or x2_col >= self.row_test_size or y1_col >= self.column_test_size or y2_col >= self.column_test_size or np.isnan(x1_col) or np.isnan(x2_col) or np.isnan(y1_col) or np.isnan(y2_col):
                        possible = False
                    if x1_col < 0 or x2_col < 0 or y1_col < 0 or y2_col < 0:
                        possible = False

                    if (self.prop == True) and (possible == True):
                        x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

        elif ((A2/A1 == A3/A2 == A4/A3) and (B2/B1 == B3/B2 == B4/B3)):
            if (B1 % A1 == 0):
                # $B=$@5$NI,MW$"$j(B

                # print("omomi 1")
                omomi_fault_num = 1

                y1 = int(B1/A1)
                d1 = -A1
                x1 = np.log2(int(A2/(-d1))) + 1

                # $BD"?,$r9g$o$;$k(B
                x1 -= 1
                y1 -= 1

                if x1 >= self.row_test_size or x2 >= self.row_test_size or y1 >= self.column_test_size or y2 >= self.column_test_size or np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
                    possible = False
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    possible = False

                if (self.prop == True) and (possible == True):
                    x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

            else:
                #fault_num = 2
                if (B1/A1 == B2/A2 == B3/A3 == B4/A4):
                    #print("omomi or kensa yoko 2")
                    x = np.log2(A2/A1)
                    # $BDs0F<jK!<B9T(B
                    if self.prop == True:
                        x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num = self.column_test(i, j, x, A1, B1)

                        self.tuuka = True

                        #print("self.W is : \n"+str(self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)]))
                        #print("self.column_check_sum is : \n"+str(self.column_check_sum[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*2):int((j*2)+2)]))
                        
                        #print("self.row_check_sum is : \n"+str(self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))

                        #print("self.fault_locate is : \n"+str(self.fault_locate))
                        #print("self.fault_locate_column_check_sum is : \n"+str(self.fault_locate_column_check_sum))
                        #print("self.fault_locate_row_check_sum.T is : \n"+str(self.fault_locate_row_check_sum.T))
                        #print("self.comp_fault_map is : \n"+str(self.comp_fault_map))
                        #print("self.comp_fault_map_column_check_sum is : \n"+str(self.comp_fault_map_column_check_sum))
                        #print("self.comp_fault_map_row_check_sum.T is : \n"+str(self.comp_fault_map_row_check_sum.T))

                        #print("A1 is : "+str(A1))
                        #print("A2 is : "+str(A2))
                        #print("A3 is : "+str(A3))
                        #print("A4 is : "+str(A4))

                        #print("B1 is : "+str(B1))
                        #print("B2 is : "+str(B2))
                        #print("B3 is : "+str(B3))
                        #print("B4 is : "+str(B4))

                        #print("x1 is : "+str(x1))
                        #print("x2 is : "+str(x2))
                        #print("y1 is : "+str(y1))
                        #print("y2 is : "+str(y2))

                        #print("x1_col is : "+str(x1_col))
                        #print("x2_col is : "+str(x2_col))
                        #print("y1_col is : "+str(y1_col))
                        #print("y2_col is : "+str(y2_col))

                        #print("x1_row is : "+str(x1_row))
                        #print("x2_row is : "+str(x2_row))
                        #print("y1_row is : "+str(y1_row))
                        #print("y2_row is : "+str(y2_row))

                        if (col_checksum_fnum == 1):
                            if (d1 == (A1)):
                                y1_col = 1
                                d1_col = d1 * y1 + B1
                                

                            else:
                                y1_col = 0
                                d1_col = d1 + A1

                        #only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                        return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num
                    
                    else:
                        possible = False
                        #only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)
                        return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num
                
                else:
                    #print("kensa 2 c")
                    only_check = True
                    check = True
                    col_checksum_fnum = 2

                    y1_col = 0
                    y2_col = 1
                    d1_col = A1
                    d2_col = B1
                    x1_col = np.log2(A2/d1_col)
                    x2_col = np.log2(B2/d2_col)

                    if x1_col >= self.row_test_size or x2_col >= self.row_test_size or y1_col >= self.column_test_size or y2_col >= self.column_test_size or np.isnan(x1_col) or np.isnan(x2_col) or np.isnan(y1_col) or np.isnan(y2_col):
                        possible = False
                    if x1_col < 0 or x2_col < 0 or y1_col < 0 or y2_col < 0:
                        possible = False

                    if (self.prop == True) and (possible == True):
                        x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num


        elif ((A2/A1 == A3/A2 == A4/A3) or (B2/B1 == B3/B2 == B4/B3)):
            if (A2/A1 == A3/A2 == A4/A3):
                # print("omomi 1 kensa 1 a")
                omomi_fault_num = 1
                col_checksum_fnum = 1
                check = True
                y1_col = 1

                if (int(B1*B3 - B2*B2)) != 0:
                    eta = (int(B1*B4 - B2*B3)) / (int(B1*B3 - B2*B2))

                    if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                        d1 = -A1

                        x1     = np.log2(np.abs(A2 / d1)) + 1
                        x1_cmp = np.floor(np.log2(eta)) + 1

                        #print("x1 is : "+str(x1))
                        #print("x1_cmp is : "+str(x1_cmp))

                        if (x1 == x1_cmp):
                            if ((eta - np.power(2.0, x1-1)) != 0):
                                x1_col = np.log2(eta - np.power(2.0, x1-1)) + 1
                                y1 = np.abs((np.power(2.0, x1_col-1)*B1 - B2) / (d1 * (np.power(2.0, x1_col-1) - np.power(2.0, x1-1))))
                                d1_col = B1 + d1 * y1

                        else:
                            x1_col = x1_cmp
                            y1 = np.abs((np.power(2.0, x1_col-1)*B1 - B2) / (d1 * (np.power(2.0, x1_col-1) - np.power(2.0, x1-1))))
                            d1_col = B1 + d1 * y1

                # $BD"?,$r9g$o$;$k(B
                x1 -= 1
                x1_col -= 1
                y1 -= 1

                #print("d1 is : "+str(d1))
                #print("y1 is : "+str(y1))
                #print("x1 is : "+str(x1))

                if x1 >= self.row_test_size or x1_col >= self.row_test_size or y1 >= self.column_test_size or y1_col >= self.column_test_size or np.isnan(x1) or np.isnan(x1_col) or np.isnan(y1) or np.isnan(y1_col):
                    possible = False
                if x1 < 0 or x1_col < 0 or y1 < 0 or y1_col < 0:
                    possible = False

                if (self.prop == True) and (possible == True):
                    x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

            else:
                # print("omomi 1 kensa 1 b")
                omomi_fault_num = 1
                col_checksum_fnum = 1
                check = True
                y1_col = 0

                if (int(A1*A3 - A2*A2)) != 0:
                    eta = (int(A1*A4 - A2*A3)) / (int(A1*A3 - A2*A2))
                    if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                        x1 = np.log2(np.abs(B2 / B1)) + 1
                        x1_cmp = np.floor(np.log2(eta)) + 1
                
                        if (x1 == x1_cmp):
                            if ((eta - np.power(2.0, x1-1)) != 0):
                                x1_col = np.log2(eta - np.power(2.0, x1-1)) + 1

                        else:
                            x1_col = x1_cmp

                        d1 = (np.power(2.0, x1_col-1)*A1 - A2) / (np.power(2.0, x1-1) - np.power(2.0, x1_col-1))
                        d1_col = (A1 + d1)

                        y1 = np.abs(B1 / d1)

                #print("d1_col is : "+str(d1_col))
                #print("d1 is : "+str(d1))
                #print("y1 is : "+str(y1))
                #print("x1 is : "+str(x1))

                # $BD"?,$r9g$o$;$k(B
                x1 -= 1
                x1_col -= 1
                y1 -= 1

                if x1 >= self.row_test_size or x1_col >= self.row_test_size or y1 >= self.column_test_size or y1_col >= self.column_test_size or np.isnan(x1) or np.isnan(x1_col) or np.isnan(x1_col) or np.isnan(y1) or np.isnan(y1_col):
                    possible = False
                if x1 < 0 or x1_col < 0 or y1 < 0 or y1_col < 0:
                    possible = False

                if (self.prop == True) and (possible == True):
                    x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

                return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

        else:
            # print("omomi 2")
            omomi_fault_num = 2

            if (int(A1*A3 - A2*A2)) != 0:
                eta = (int(A1*A4 - A2*A3)) / (int(A1*A3 - A2*A2))

                if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                    x1 = np.floor(np.log2(eta)) + 1
                    if ((eta - np.power(2.0, x1-1)) != 0):
                        x2 = np.log2(eta - np.power(2.0, x1-1)) + 1

                        d1 = -(np.power(2.0, x2-1)*A1 - A2) / (np.power(2.0, x2-1) - np.power(2.0, x1-1))
                        d2 = -(A1 + d1) #* -1

                        y1 = (np.power(2.0, x2-1)*B1 - B2) / (-d1 * (np.power(2.0, x1-1) - np.power(2.0, x2-1)))
                        y2 = (B1 + (y1*-d1)) / -d2

                        y1 = y1 * (-1)

            # $BD"?,$r9g$o$;$k(B
            x1 -= 1
            x2 -= 1
            y1 -= 1
            y2 -= 1

            y1 = np.round(y1)
            y2 = np.round(y2)

            if x1 >= self.row_test_size or x2 >= self.row_test_size or y1 >= self.column_test_size or y2 >= self.column_test_size or np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
                possible = False
            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                possible = False

            if (self.prop == True) and (possible == True):
                x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free = self.only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

            return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

        #if prop == True:
        #    only_column_test(i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num)

        return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, fault_free, omomi_fault_num

    def column_test(self, i, j, x, A1, B1):

        self.yoko = True

        row_check      = False
        row_only_check = False
        # possible   = True
        # fault_free = False
        row_checksum_fnum = 0 
        col_checksum_fnum = 0

        omomi_1    = False
        check      = False
        only_check = False
        possible   = True
        omomi_fault_num  = 0

        eta = 0
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        d1 = 0
        d2 = 0

        x1_row = 0
        x2_row = 0
        y1_row = 0
        y2_row = 0
        d1_row = 0
        d2_row = 0

        x1_col = 0
        x2_col = 0
        y1_col = 0
        y2_col = 0
        d1_col = 0
        d2_col = 0

        # tuika for row amari
        tfra = 0
        # tuika for column amari
        tfca = 0
        
        if (self.row_amari1 == True and i == self.per_row-1):
            tfra = 1

        if (self.column_amari1 == True and j == self.per_column-1):
            tfca = 1

        test_vector = np.array([np.array([pow(2, k*0) for k in range(self.column_test_size)]),
                                np.array([pow(2, k*1) for k in range(self.column_test_size)]),
                                np.array([pow(2, k*2) for k in range(self.column_test_size)]),
                                np.array([pow(2, k*3) for k in range(self.column_test_size)])])

        test_vector_amari1 = np.array([np.array([pow(2, k*0) for k in range(self.column_test_size+1)]),
                                       np.array([pow(2, k*1) for k in range(self.column_test_size+1)]),
                                       np.array([pow(2, k*2) for k in range(self.column_test_size+1)]),
                                       np.array([pow(2, k*3) for k in range(self.column_test_size+1)])])

        if self.column_amari1 == False and self.row_amari1 == False:
            if (self.column_size % self.column_test_size != 0) and (j == self.per_column-1):
                O1     = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O1_sum = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2     = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2_sum = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3     = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3_sum = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4     = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4_sum = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
            #print("self.W is : "+str(self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            #print("self.row_check_sum is : "+str(self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            else:
                O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        elif self.column_amari1 == True and self.row_amari1 == False:
            if (j == self.per_column-1):
                O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O1_sum = np.dot(test_vector_amari1[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O2_sum = np.dot(test_vector_amari1[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O3_sum = np.dot(test_vector_amari1[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O4_sum = np.dot(test_vector_amari1[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
            #print("self.W is : "+str(self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            #print("self.row_check_sum is : "+str(self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            else:
                O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        elif self.column_amari1 == False and self.row_amari1 == True:
            if (self.column_size % self.column_test_size != 0) and (j == self.per_column-1):
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                else:
                    O1     = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

            else:
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                else:
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        elif self.column_amari1 == True and self.row_amari1 == True:
            if (j == self.per_column-1):
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O1_sum = np.dot(test_vector_amari1[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2_sum = np.dot(test_vector_amari1[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3_sum = np.dot(test_vector_amari1[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4_sum = np.dot(test_vector_amari1[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                else:
                    O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O1_sum = np.dot(test_vector_amari1[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2_sum = np.dot(test_vector_amari1[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3_sum = np.dot(test_vector_amari1[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4_sum = np.dot(test_vector_amari1[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)

            else:
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                else:
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        WG = np.arange(1, O1.shape[0]+1)

        #print("self.column_check_sum is : "+str(self.column_check_sum.shape))
        #print("self.row_check_sum is : "+str(self.row_check_sum.shape))

        #print("i is : "+str(i))
        #print("j is : "+str(j))
        #print("self.row_check_sum.T is : "+str(self.row_check_sum.T[int(i*self.column_test_size):int((i+1)*self.column_test_size), int(j*2):int((j*2)+2)]))
        #print("O1 is : "+str(O1))
        #print("O1_sum is : "+str(O1_sum))

        C1 = np.round(np.sum(O1) - O1_sum[0], decimals=6)
        C2 = np.round(np.sum(O2) - O2_sum[0], decimals=6)
        C3 = np.round(np.sum(O3) - O3_sum[0], decimals=6)
        C4 = np.round(np.sum(O4) - O4_sum[0], decimals=6)

        D1 = np.round(np.sum(np.dot(WG, O1)) - O1_sum[1], decimals=6)
        D2 = np.round(np.sum(np.dot(WG, O2)) - O2_sum[1], decimals=6)
        D3 = np.round(np.sum(np.dot(WG, O3)) - O3_sum[1], decimals=6)
        D4 = np.round(np.sum(np.dot(WG, O4)) - O4_sum[1], decimals=6)

        # print("O1 is : "+str(O1))
        # print("O2 is : "+str(O2))
        # print("O3 is : "+str(O3))
        # print("O4 is : "+str(O4))

        #print("C1 is : "+str(C1))
        #print("C2 is : "+str(C2))
        #print("C3 is : "+str(C3))
        #print("C4 is : "+str(C4))

        #print("D1 is : "+str(D1))
        #print("D2 is : "+str(D2))
        #print("D3 is : "+str(D3))
        #print("D4 is : "+str(D4))

        if int(C1 == C2 == C3 == C4 == 0) and int(D1 == D2 == D3 == D4 == 0):
            #print("row fault_free")
            col_checksum_fnum = 2
            y1_col = 0
            y2_col = 1
            x1_col = x
            x2_col = x

            return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

        if (int(C1 == C2 == C3 == C4 == 0) or int(D1 == D2 == D3 == D4 == 0)):
            col_checksum_fnum = 2

            y1_col = 0
            y2_col = 1
            x1_col = x
            x2_col = x
            d1_col = A1
            d2_col = B1

            check = True
            only_check = True

            if(int(C1 == C2 == C3 == C4 == 0)):
                if (D1*D4 - D2*D3 == 0) and (D1*D3 - D2*D2 == 0):
                    #print("row kensa 1 a")
                    row_checksum_fnum = 1
                    x1_row = 1
                    y1_row = int(np.log2(int(D2/D1)) + 1)
                    d1_row = D1

                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1

                    if x1_row >= self.row_test_size + tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

                else:
                    #print("row kensa 2 a")
                    # $B4pK\E*$K$OL5M}$@$1$I!$%F%9%H%5%$%:$r(B2$B$K$7$?$H$-$@$18!=P2DG=(B
                    # possible = False
                    row_checksum_fnum = 2
                    x1_row = 1
                    x2_row = 1

                    if (int(D1*D3 - D2*D2)) != 0:
                        eta = int((D1*D4 - D2*D3)) / int((D1*D3 - D2*D2))
                        if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                            y1_row = int(np.floor(np.log2(eta)) + 1)
                            # $B9TNs@8@.;~$K(Bcheck sum$BJ,$1$F$k$+$i!$$"$H$GD{@5$9$kI,MW$"$j(B($B7W;;MQ$K0l;~E*$KBeF~$7$F7W;;$9$k(B)$B$H!$;W$C$?$,!$(Bx2$B!J9T$N0LCV!K$J$i!$FC$KLdBj$J$+$C$?(B
                            if ((eta - np.power(2.0, y1_row-1)) != 0):
                                y2_row = int(np.log2(eta - np.power(2, y1_row-1))) + 1
                                d1_row = (-D2 + int(np.power(2.0, y2_row-1)*D1)) / (int(np.power(2.0, y2_row-1) - np.power(2.0, y1_row-1)))
                                d2_row = D1 - d1_row

                    # both = True
                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1
                    y2_row -= 1

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

            elif(int(D1 == D2 == D3 == D4 == 0)):

                if (int(C1*C4 - C2*C3) == 0) and (int(C1*C3 - C2*C2) == 0):
                    #print("row kensa 1 b")
                    row_checksum_fnum = 1
                    x1_row = 0
                    y1_row = np.log2(int(C2/C1)) + 1
                    d1_row = C1

                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1

                    both = True

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

                else:
                    #print("row kensa 2 b")
                    # possible = False
                    row_checksum_fnum = 2
                    x1_row = 0
                    x2_row = 0

                    if (int(C1*C3 - C2*C2)) != 0:
                        eta = int((C1*C4 - C2*C3)) / int((C1*C3 - C2*C2))

                        if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                            y1_row = int(np.floor(np.log2(eta)) + 1)
                            # $B9TNs@8@.;~$K(Bcheck sum$BJ,$1$F$k$+$i!$$"$H$GD{@5$9$kI,MW$"$j(B($B7W;;MQ$K0l;~E*$KBeF~$7$F7W;;$9$k(B)$B$H!$;W$C$?$,!$(Bx2$B!J9T$N0LCV!K$J$i!$FC$KLdBj$J$+$C$?(B
                            if ((eta - np.power(2.0, y1_row-1)) != 0):
                                y2_row = int(np.log2(eta - np.power(2, y1_row-1))) + 1
                                d1_row = (-C2 + int(np.power(2.0, y2_row-1)*C1)) / (int(np.power(2.0, y2_row-1) - np.power(2.0, y1_row-1)))
                                d2_row = C1 - d1_row
                        
                    #both = True
                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1
                    y2_row -= 1

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num
        
        elif ((C2/C1 == C3/C2 == C4/C3) and (D2/D1 == D3/D2 == D4/D3)):
            if (D1 % C1 == 0):

                #print("row omomi 1")
                omomi_1 = True
                check = True
                omomi_fault_num = 1
                col_checksum_fnum = 1
                row_checksum_fnum = 2

                # x$B$H(By$B$rF~$lBX$($F7W;;(B
                x1 = int(D1/C1)
                d1 = -C1
                y1 = np.log2(int(C2/-d1)) + 1

                # $BD"?,$r9g$o$;$k(B
                x1 -= 1
                y1 -= 1

                x1_row = 0
                x2_row = 1
                y1_row = y1
                y2_row = y1

                y1_col = 0
                y2_col = 1
                x1_col = x
                x2_col = x

                if x1 >= self.row_test_size+tfra or x2 >= self.row_test_size+tfra or y1 >= self.column_test_size+tfca or y2 >= self.column_test_size+tfca or np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
                    possible = False
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    possible = False

                return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

            else:
                if (D1/C1 == D2/C2 == D3/C3 == D4/C4):
                    #print("row omomi or kensa yoko 2")
                    # omomi$B$N8N>c2U=j$OFCDj2DG=(B. $B$=$NB>$OIT2D(B. 
                    # $BDI5-!%<B$O$3$N8N>c$bFCDj2DG=!%(B
                    possible = False
                    #row_check = True
                    #row_only_check = True
                    #row_checksum_fnum = 2
                    #check = True
                    #omomi_fault_num = 1
                    #col_checksum_fnum = 2
                    #y1 = np.log2(C2/C1)

                    #y1_row = 0
                    #y2_row = 1
                    #x1_row = x
                    #x2_row = x

                    #y1_col = 0
                    #y2_col = 1
                    #x1_col = x
                    #x2_col = x

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

                else:
                    #print("row kensa 2 c")
                    col_checksum_fnum = 2
                    row_checksum_fnum = 2
                    row_only_check    = True
                    row_check         = True

                    y1_col = 0
                    y2_col = 1
                    x1_col = x
                    x2_col = x
                    d1_col = A1
                    d2_col = B1

                    x1_row = 0
                    x2_row = 1
                    d1_row = C1
                    d2_row = D1
                    y1_row = np.log2(C2/d1_row)
                    y2_row = np.log2(D2/d2_row)

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

        elif ((C2/C1 == C3/C2 == C4/C3) or (D2/D1 == D3/D2 == D4/D3)):
            if (C2/C1 == C3/C2 == C4/C3):
                #print("row omomi 1 kensa 1 a")
                omomi_1 = True
                col_checksum_fnum = 1
                row_checksum_fnum = 1
                omomi_fault_num = 1
                check = True
                row_check = True

                x1_row = 1

                if (int(D1*D3 - D2*D2)) != 0:
                    eta = (int(D1*D4 - D2*D3)) / (int(D1*D3 - D2*D2))
                    if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                        d1 = -C1

                        y1     = np.log2(np.abs(C2 / d1)) + 1
                        y1_cmp = np.floor(np.log2(eta)) + 1

                        #print("y1 is : "+str(x1))
                        #print("y1_cmp is : "+str(y1_cmp))

                        if (y1 == y1_cmp):
                            if ((eta - np.power(2.0, y1-1)) != 0):
                                y1_row = np.log2(eta - np.power(2.0, y1-1)) + 1

                        else:
                            y1_row = y1_cmp

                        x1 = np.abs((np.power(2.0, y1_row-1)*D1 - D2) / (d1 * (np.power(2.0, y1_row-1) - np.power(2.0, y1-1))))
                        d1_row = D1 + d1 * x1

                # $BD"?,$r9g$o$;$k(B
                y1 -= 1
                y1_row -= 1
                x1 -= 1
                #y2 -= 1

                #print("d1 is : "+str(d1))
                #print("y1 is : "+str(y1))
                #print("x1 is : "+str(x1))

                y1_col = 0
                y2_col = 1
                x1_col = x
                x2_col = x

                if x1 >= self.row_test_size+tfra or x1_row >= self.row_test_size+tfra or y1 >= self.column_test_size+tfca or y1_row >= self.column_test_size+tfca or np.isnan(y1_row) or np.isnan(x1_row) or np.isnan(x1) or np.isnan(y1):
                    possible = False
                if x1 < 0 or x1_row < 0 or y1 < 0 or y1_row < 0:
                    possible = False

                return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

            else:
                # print("row omomi 1 kensa 1 b")
                omomi_1 = True
                col_checksum_fnum = 1
                row_checksum_fnum = 1
                omomi_fault_num = 1
                check = True
                row_check = True

                x1_row = 0
                #d1_col = B2 - A1

                if (int(C1*C3 - C2*C2)) != 0:
                    eta = (int(C1*C4 - C2*C3)) / (int(C1*C3 - C2*C2))
                    #print("eta is : "+str(eta))

                    if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                        y1 = np.log2(np.abs(D2 / D1)) + 1
                        #print("y1 is : "+str(y1))
                        y1_cmp = np.floor(np.log2(eta)) + 1

                        if (y1 == y1_cmp):
                            if ((eta - np.power(2.0, y1-1)) != 0):
                                y1_row = np.log2(eta - np.power(2.0, y1-1)) + 1

                        else:
                            y1_row = y1_cmp

                        d1 = (np.power(2.0, y1_row-1)*C1 - C2) / (np.power(2.0, y1-1) - np.power(2.0, y1_row-1))
                        d1_row = (C1 + d1)

                        x1 = np.abs(D1 / d1)

                #print("d1_row is : "+str(d1_row))
                #print("d1 is : "+str(d1))
                #print("y1 is : "+str(y1))
                #print("x1 is : "+str(x1))
                
                # $BD"?,$r9g$o$;$k(B
                y1 -= 1
                y1_row -= 1
                x1 -= 1

                y1_col = 0
                y2_col = 1
                x1_col = x
                x2_col = x

                #print("x1_row is : "+str(x1_row))
                #print("x2_row is : "+str(x2_row))
                #print("y1_row is : "+str(y1_row))
                #print("y2_row is : "+str(y2_row))

                if x1 >= self.row_test_size+tfra or x1_row >= self.row_test_size+tfra or y1 >= self.column_test_size+tfca or y1_row >= self.column_test_size+tfca or np.isnan(y1_row) or np.isnan(x1_row) or np.isnan(x1) or np.isnan(y1):
                    possible = False
                if x1 < 0 or x1_row < 0 or y1 < 0 or y1_row < 0:
                    possible = False

                return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

        else:
            #print("row omomi 2")
            omomi_fault_num = 2

            if (int(C1*C3 - C2*C2)) != 0:
                eta = (C1*C4 - C2*C3) / (C1*C3 - C2*C2)
                if ((not np.isnan(eta)) and eta >= 2):

                    y1 = np.floor(np.log2(eta)) + 1
                    if ((eta - np.power(2.0, y1-1)) != 0):
                        y2 = np.log2(eta - np.power(2.0, y1-1)) + 1

                        d1 = (np.power(2.0, y2-1)*C1 - C2) / (np.power(2.0, y2-1) - np.power(2.0, y1-1))
                        d2 = (C1 - d1) #* -1

                        x1 = (np.power(2.0, y2-1)*D1 - D2) / (d1 * (np.power(2.0, y1-1) - np.power(2.0, y2-1)))
                        x2 = (D1 + (x1*d1)) / d2

                        x1 = np.round(x1 * (-1))

            # $BD"?,$r9g$o$;$k(B
            x1 -= 1
            x2 -= 1
            y1 -= 1
            y2 -= 1
            d1 *= -1
            d2 *= -1

            if x1 >= self.row_test_size+tfra or x2 >= self.row_test_size+tfra or y1 >= self.column_test_size+tfca or y2 >= self.column_test_size+tfca or np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2):
                possible = False
            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                possible = False

            return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

        return x1, x2, y1, y2, x1_col, x2_col, y1_col, y2_col, x1_row, x2_row, y1_row, y2_row, d1, d2, d1_col, d2_col, d1_row, d2_row, col_checksum_fnum, row_checksum_fnum, possible, omomi_fault_num

    def only_column_test(self, i, j, x1, x2, y1, y2, d1, d2, omomi_fault_num):

        # tuika for row amari
        tfra = 0
        # tuika for column amari
        tfca = 0
        
        if (self.row_amari1 == True and i == self.per_row-1):
            tfra = 1

        if (self.column_amari1 == True and j == self.per_column-1):
            tfca = 1

        row_checksum_fnum = 0 
        
        possible   = True
        fault_free = False

        eta = 0
        x1 = x1
        x2 = x2
        y1 = y1
        y2 = y2
        d1 = d1
        d2 = d2

        x1_row = 0
        x2_row = 0
        y1_row = 0
        y2_row = 0
        d1_row = 0
        d2_row = 0

        test_vector = np.array([np.array([pow(2, k*0) for k in range(self.column_test_size)]),
                                np.array([pow(2, k*1) for k in range(self.column_test_size)]),
                                np.array([pow(2, k*2) for k in range(self.column_test_size)]),
                                np.array([pow(2, k*3) for k in range(self.column_test_size)])])

        test_vector_amari1 = np.array([np.array([pow(2, k*0) for k in range(self.column_test_size+1)]),
                                       np.array([pow(2, k*1) for k in range(self.column_test_size+1)]),
                                       np.array([pow(2, k*2) for k in range(self.column_test_size+1)]),
                                       np.array([pow(2, k*3) for k in range(self.column_test_size+1)])])

        if self.column_amari1 == False and self.row_amari1 == False:
            if (self.column_size % self.column_test_size != 0) and (j == self.per_column-1):
                O1     = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O1_sum = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2     = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2_sum = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3     = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3_sum = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4     = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4_sum = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
            #print("self.W is : "+str(self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            #print("self.row_check_sum is : "+str(self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            else:
                O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        elif self.column_amari1 == True and self.row_amari1 == False:
            if (j == self.per_column-1):
                O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O1_sum = np.dot(test_vector_amari1[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O2_sum = np.dot(test_vector_amari1[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O3_sum = np.dot(test_vector_amari1[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                O4_sum = np.dot(test_vector_amari1[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
            #print("self.W is : "+str(self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            #print("self.row_check_sum is : "+str(self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T))
            else:
                O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        elif self.column_amari1 == False and self.row_amari1 == True:
            if (self.column_size % self.column_test_size != 0) and (j == self.per_column-1):
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                else:
                    O1     = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:int(self.column_size % self.column_test_size)], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

            else:
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                else:
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        elif self.column_amari1 == True and self.row_amari1 == True:
            if (j == self.per_column-1):
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O1_sum = np.dot(test_vector_amari1[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2_sum = np.dot(test_vector_amari1[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3_sum = np.dot(test_vector_amari1[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4_sum = np.dot(test_vector_amari1[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                else:
                    O1     = np.dot(test_vector_amari1[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O1_sum = np.dot(test_vector_amari1[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2     = np.dot(test_vector_amari1[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O2_sum = np.dot(test_vector_amari1[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3     = np.dot(test_vector_amari1[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O3_sum = np.dot(test_vector_amari1[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4     = np.dot(test_vector_amari1[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)
                    O4_sum = np.dot(test_vector_amari1[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size+1)].T)

            else:
                if (j == self.per_row-1):
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size+1), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                else:
                    O1     = np.dot(test_vector[0,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O1_sum = np.dot(test_vector[0,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2     = np.dot(test_vector[1,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O2_sum = np.dot(test_vector[1,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3     = np.dot(test_vector[2,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O3_sum = np.dot(test_vector[2,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4     = np.dot(test_vector[3,:], self.W[int(i*self.row_test_size):int((i+1)*self.row_test_size), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)
                    O4_sum = np.dot(test_vector[3,:], self.row_check_sum[int(i*2):int((i*2)+2), int(j*self.column_test_size):int((j+1)*self.column_test_size)].T)

        WG = np.arange(1, O1.shape[0]+1)

        #print("self.column_check_sum is : "+str(self.column_check_sum.shape))
        #print("self.row_check_sum is : "+str(self.row_check_sum.shape))

        #print("i is : "+str(i))
        #print("j is : "+str(j))
        #print("self.row_check_sum.T is : "+str(self.row_check_sum.T[int(i*self.column_test_size):int((i+1)*self.column_test_size), int(j*2):int((j*2)+2)]))
        #print("O1 is : "+str(O1))
        #print("O1_sum is : "+str(O1_sum))

        C1 = np.round(np.sum(O1) - O1_sum[0], decimals=6)
        C2 = np.round(np.sum(O2) - O2_sum[0], decimals=6)
        C3 = np.round(np.sum(O3) - O3_sum[0], decimals=6)
        C4 = np.round(np.sum(O4) - O4_sum[0], decimals=6)

        D1 = np.round(np.sum(np.dot(WG, O1)) - O1_sum[1], decimals=6)
        D2 = np.round(np.sum(np.dot(WG, O2)) - O2_sum[1], decimals=6)
        D3 = np.round(np.sum(np.dot(WG, O3)) - O3_sum[1], decimals=6)
        D4 = np.round(np.sum(np.dot(WG, O4)) - O4_sum[1], decimals=6)

        # print("O1 is : "+str(O1))
        # print("O2 is : "+str(O2))
        # print("O3 is : "+str(O3))
        # print("O4 is : "+str(O4))

        # print("C1 is : "+str(C1))
        # print("C2 is : "+str(C2))
        # print("C3 is : "+str(C3))
        # print("C4 is : "+str(C4))

        # print("D1 is : "+str(D1))
        # print("D2 is : "+str(D2))
        # print("D3 is : "+str(D3))
        # print("D4 is : "+str(D4))

        # C$B$+$i(Bd1, d2$B$NCM$r8:;;$9$k(B
        C1 = np.round(C1 + (d1 * pow(2,0*y1)) + (d2 * pow(2,0*y2)), decimals=6)
        C2 = np.round(C2 + (d1 * pow(2,1*y1)) + (d2 * pow(2,1*y2)), decimals=6)
        C3 = np.round(C3 + (d1 * pow(2,2*y1)) + (d2 * pow(2,2*y2)), decimals=6)
        C4 = np.round(C4 + (d1 * pow(2,3*y1)) + (d2 * pow(2,3*y2)), decimals=6)

        # $BNs(B(x)$B$NCM$r9MN8$7$F!$(BD$B$+$i(Bd1, d2$B$NCM$r8:;;$9$k(B
        D1 = np.round(D1 + d1 * (x1+1) * pow(2,0*y1) + d2 * (x2+1) * pow(2,0*y2), decimals=6)
        D2 = np.round(D2 + d1 * (x1+1) * pow(2,1*y1) + d2 * (x2+1) * pow(2,1*y2), decimals=6)
        D3 = np.round(D3 + d1 * (x1+1) * pow(2,2*y1) + d2 * (x2+1) * pow(2,2*y2), decimals=6)
        D4 = np.round(D4 + d1 * (x1+1) * pow(2,3*y1) + d2 * (x2+1) * pow(2,3*y2), decimals=6)

        #print("C1 (after) is : "+str(C1))
        #print("C2 (after) is : "+str(C2))
        #print("C3 (after) is : "+str(C3))
        #print("C4 (after) is : "+str(C4))
 
        #print("D1 (after) is : "+str(D1))
        #print("D2 (after) is : "+str(D2))
        #print("D3 (after) is : "+str(D3))
        #print("D4 (after) is : "+str(D4))

        if int(C1 == C2 == C3 == C4 == 0) and int(D1 == D2 == D3 == D4 == 0):
            # print("only row fault_free")
            row_checksum_fnum = 0

            return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

        if (int(C1 == C2 == C3 == C4 == 0) or int(D1 == D2 == D3 == D4 == 0)):
            if(int(C1 == C2 == C3 == C4 == 0)):
                if (D1*D4 - D2*D3 == 0) and (D1*D3 - D2*D2 == 0):
                    # print("only row kensa 1 a")
                    row_checksum_fnum = 1
                    x1_row = 1
                    y1_row = int(np.log2(int(D2/D1)) + 1)
                    d1_row = D1

                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

                else:
                    # print("only row kensa 2 a")
                    # $B4pK\E*$K$OL5M}$@$1$I!$%F%9%H%5%$%:$r(B2$B$K$7$?$H$-$@$18!=P2DG=(B
                    # possible = False
                    row_checksum_fnum = 2
                    x1_row = 1
                    x2_row = 1

                    if (int(D1*D3 - D2*D2)) != 0:
                        eta = int((D1*D4 - D2*D3)) / int((D1*D3 - D2*D2))
                        if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                            y1_row = int(np.floor(np.log2(eta)) + 1)
                            # $B9TNs@8@.;~$K(Bcheck sum$BJ,$1$F$k$+$i!$$"$H$GD{@5$9$kI,MW$"$j(B($B7W;;MQ$K0l;~E*$KBeF~$7$F7W;;$9$k(B)$B$H!$;W$C$?$,!$(Bx2$B!J9T$N0LCV!K$J$i!$FC$KLdBj$J$+$C$?(B
                            if ((eta - np.power(2.0, y1_row-1)) != 0):
                                y2_row = int(np.log2(eta - np.power(2, y1_row-1))) + 1
                                d1_row = (-D2 + int(np.power(2.0, y2_row-1)*D1)) / (int(np.power(2.0, y2_row-1) - np.power(2.0, y1_row-1)))
                                d2_row = D1 - d1_row

                    # both = True
                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1
                    y2_row -= 1

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

            elif(int(D1 == D2 == D3 == D4 == 0)):

                if (int(C1*C4 - C2*C3) == 0) and (int(C1*C3 - C2*C2) == 0):
                    # print("only row kensa 1 b")
                    row_checksum_fnum = 1
                    x1_row = 0
                    y1_row = np.log2(int(C2/C1)) + 1
                    d1_row = C1

                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1

                    both = True

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

                else:
                    # print("only row kensa 2 b")
                    # possible = False
                    row_checksum_fnum = 2
                    x1_row = 0
                    x2_row = 0

                    if (int(C1*C3 - C2*C2)) != 0:
                        eta = int((C1*C4 - C2*C3)) / int((C1*C3 - C2*C2))

                        if ((not np.isnan(eta)) and (not np.isinf(eta)) and eta >= 2):
                            y1_row = int(np.floor(np.log2(eta)) + 1)
                            # $B9TNs@8@.;~$K(Bcheck sum$BJ,$1$F$k$+$i!$$"$H$GD{@5$9$kI,MW$"$j(B($B7W;;MQ$K0l;~E*$KBeF~$7$F7W;;$9$k(B)$B$H!$;W$C$?$,!$(Bx2$B!J9T$N0LCV!K$J$i!$FC$KLdBj$J$+$C$?(B
                            if ((eta - np.power(2.0, y1_row-1)) != 0):
                                y2_row = int(np.log2(eta - np.power(2, y1_row-1))) + 1
                                d1_row = (-C2 + int(np.power(2.0, y2_row-1)*C1)) / (int(np.power(2.0, y2_row-1) - np.power(2.0, y1_row-1)))
                                d2_row = C1 - d1_row
                        
                    #both = True
                    # $BD"?,$r9g$o$;$k(B
                    y1_row -= 1
                    y2_row -= 1

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free
        
        elif ((C2/C1 == C3/C2 == C4/C3) and (D2/D1 == D3/D2 == D4/D3)):
            if (D1 % C1 == 0):
                row_checksum_fnum = 0
                return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

            else:
                if (D1/C1 == D2/C2 == D3/C3 == D4/C4):
                    # print("only row kensa yoko 2")
                    # omomi$B$N8N>c2U=j$OFCDj2DG=(B. $B$=$NB>$OIT2D(B. 
                    # $BDI5-!%<B$O$3$N8N>c$bFCDj2DG=!%(B
                    # $BDIDI5-!%$d$C$Q$jL5M}!%(B
                    # possible = False
                    row_checksum_fnum = 0

                    #x1_row = 0
                    #x2_row = 1
                    #x1_row = x
                    #x2_row = x

                    return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

                else:
                    # print("only row kensa 2 c")
                    row_checksum_fnum = 2

                    x1_row = 0
                    x2_row = 1
                    d1_row = C1
                    d2_row = D1
                    y1_row = np.log2(C2/d1_row)
                    y2_row = np.log2(D2/d2_row)

                    if x1_row >= self.row_test_size+tfra or x2_row >= self.row_test_size+tfra or y1_row >= self.column_test_size+tfca or y2_row >= self.column_test_size+tfca or np.isnan(x1_row) or np.isnan(x2_row) or np.isnan(y1_row) or np.isnan(y2_row):
                        possible = False
                    if x1_row < 0 or x2_row < 0 or y1_row < 0 or y2_row < 0:
                        possible = False

                    return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

        else:
            # $BFf$N%Q%?!<%s!%$3$3$NJ,4t$KMh$k8N>c$OBP>]$H$7$F$$$k8N>c$G$O$J$$!%(B
            # print("nazo")
            row_checksum_fnum = 0
            return x1_row, x2_row, y1_row, y2_row, d1_row, d2_row, row_checksum_fnum, possible, fault_free

