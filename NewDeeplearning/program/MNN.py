
import sys, os
sys.path.append(os.pardir)

import numpy as np
import gc
import math
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from program.layer_net import MNN
import random
from common.optimizer import SGD, Adam, AdaGrad
#import seaborn as sns
import sklearn.linear_model as lm
import scipy.fftpack as sp
import scipy.io
from scipy import stats
#import pandas as pd
from numba import jit
# TensorFlow and tf.keras
#import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
import time
import copy

# ---------------各種パラメータの設定．基本的にはここをいじるだけで実験可能-----------------------

# use_dataset : どのデータセットを使うのか. DIGITS, MNIST, FashionMNIST, IRISには対応．
use_dataset = "MNIST" # DIGITS, MNIST, FashionMNIST, IRIS

# activation_function : 活性化関数．Relu以外に，Sigmoid, Tanhに対応．
activation_function = "Relu" # "Sigmoid", "Tanh"

# hidden_size : 中間層の数とサイズ．numpy array形式で渡す．例えば，784-300-200-10のニューラルネットワークなら，np.array([300,200])のように設定．
# 入力層と出力層は，データセットによって自動で決定されるようになっている．
hidden_size  = np.array([100])

# フォルダ生成用に，中間層のサイズを文字列にして生成
hidden_name = str(hidden_size[0])
if hidden_size.shape[0] > 1:
    for i in range(1,hidden_size.shape[0]):
        hidden_name += "_" + str(hidden_size[i])

# optimizer : 最適化アルゴリズム．Adam, AdaGrad, SGDが選択可．
optimizer_name = "SGD" # Adam, AdaGrad

# batch_norm : Batch Normalizationを使うか使わないか．VLD原稿に出した内容の実験では利用していない．
batch_norm = False

# tau : 何回の書き込み毎にテストをするのか
tau = 5000

# mu  : メモリスタの寿命の平均
mu  = 2.5 * pow(10, 5)

# std : メモリスタの寿命の標準偏差
std = 8.0 * pow(10, 4)

# row_test_size : テストブロックの行サイズ．
row_test_size = 2

# col_test_size : テストブロックの列サイズ．
col_test_size = 2

# soft_fault_rate : クロスバーアレイ中の全メモリスタのうち，過渡故障が発生するメモリスタの割合
soft_fault_rate = 0.02

# seed : 乱数シード．適当に設定してください．
seed1 = 100
seed2 = 10000

# mini-batch-size : ミニバッチ学習時のデータサイズ．
mini_batch_size = 300

# ----------------------ここまで---------------------------

# network_size : 重み行列の数．中間層の数+1になる．中間層の数を決めれば自動で決定．
network_size = 1 + hidden_size.shape[0]

# これはとりあえず0で
test_epoch = 0

# digit と MNIST は，プログラムに対応済み．
# IRIS と Fashion-MNISTはプログラムにまだ対応してません．

if use_dataset == "DIGITS":
    # Digits : MNISTより小さい手書き文字のやつ
    # 全データ数1797枚．適当に訓練データとテストデータに分けて利用してください．
    digits = datasets.load_digits()
    np.random.seed(1)
    # 元データを一回バラバラに並び替える．その後，前半数枚を訓練データに利用．
    choice_num = np.random.permutation(digits.data.shape[0])

    x_train = digits.data[choice_num][:1347] / 16
    t_train = digits.target[choice_num][:1347]

    x_test = digits.data[choice_num][1347:1796] / 16
    t_test = digits.target[choice_num][1347:1796]

    MNN_array = np.array([64])
    MNN_array = np.concatenate([MNN_array, hidden_size])
    MNN_array = np.concatenate([MNN_array, np.array([10])])

    network_FF       = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=0.0, hard_fault=False, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=False)
    network_baseline = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=True, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_xabft    = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_prop     = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=True, seed1=seed1, seed2=seed2, fault=True)

    batch_size = mini_batch_size
    batch_loop = int(np.ceil((digits.data.shape[0] / mini_batch_size))) # とは
    test_epoch = int(10000/batch_loop)

elif use_dataset == "MNIST":
    # MNIST:データの読み込み
    # 全訓練データ数60,000枚
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    MNN_array = np.array([784])
    MNN_array = np.concatenate([MNN_array, hidden_size])
    MNN_array = np.concatenate([MNN_array, np.array([10])])
    print("MNN_array is :"+str(MNN_array))

    network_FF       = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=0.0, hard_fault=False, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=False)
    network_baseline = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=True, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_xabft    = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_prop     = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=True, seed1=seed1, seed2=seed2, fault=True)

    batch_size = mini_batch_size
    batch_loop = int(np.ceil((60000 / mini_batch_size)))
    #test_epoch = int(10000/batch_loop)

elif use_dataset == "FashionMNIST":
    # Fashion-MNIST:データの読み込み
    # 全訓練データ数60,000枚．
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test.astype('float32') / 255

    MNN_array = np.array([784])
    MNN_array = np.concatenate([MNN_array, hidden_size])
    MNN_array = np.concatenate([MNN_array, np.array([10])])

    network_FF       = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=0.0, hard_fault=False, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=False)
    network_baseline = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=True, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_xabft    = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_prop     = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=True, seed1=seed1, seed2=seed2, fault=True)

    batch_size = mini_batch_size
    batch_loop = int(np.ceil((60000 / mini_batch_size)))
    #test_epoch = int(10000/batch_loop)

elif use_dataset == "IRIS":
    # 全データ数150．適当に訓練データとテストデータに分けて利用してください．
    df = np.loadtxt("../dataset/iris.data", delimiter=",")
    #df = np.asarray(df)

    #print("df is : "+str(df))

    df = np.array(df, dtype=int)

    t = copy.deepcopy(df[:,4])
    x = copy.deepcopy(df[:,:4])

    #print("t is : "+str(t))
    #print("x is : "+str(x))

    train_data_num = 100
    test_data_num  = 150 - int(train_data_num)

    seto = 33
    vers = 33
    verg = 34

    x_train = np.zeros((train_data_num, 4))
    x_test  = np.zeros((test_data_num, 4))

    t_train = np.empty(train_data_num)
    t_test  = np.empty(test_data_num)

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    # IRISのデータセットは50区切りでそれぞれの花のデータに分かれている。
    # 最初の50がsetona, 次の50がversicolor, 最後の50がverginica

    # 下記のプログラムでは，最初の35個のデータを訓練用に、最後の15個をテストデータ利用
    x_train[:int(seto),:4],x_train[int(seto):int(seto+vers),:4],x_train[int(seto+vers):,:4] = x[:int(seto),:4],x[50:int(50+vers),:4],x[100:int(100+verg),:4]
    x_test[:int(50-seto),:4],x_test[int(50-seto):int((50-seto)+(50-vers)),:4],x_test[int((50-seto)+(50-vers)):,:4] = x[int(seto):50,:4],x[int(50+vers):100,:4],x[int(100+verg):150,:4]

    t_train[:int(seto)],t_train[int(seto):int(seto+vers)],t_train[int(seto+vers):] = t[:int(seto)],t[50:int(50+vers)],t[100:int(100+verg)]
    t_test[:int(50-seto)],t_test[int(50-seto):int((50-seto)+(50-vers))],t_test[int((50-seto)+(50-vers)):] = t[int(seto):50],t[int(50+vers):100],t[int(100+verg):150]

    x_train = np.array(x_train, dtype=int)
    x_test  = np.array(x_test, dtype=int)
    t_train = np.array(t_train, dtype=int)
    t_test  = np.array(t_test, dtype=int)

    MNN_array = np.array([4])
    MNN_array = np.concatenate([MNN_array, hidden_size])
    MNN_array = np.concatenate([MNN_array, np.array([3])])

    network_FF       = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=0.0, hard_fault=False, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=False)
    network_baseline = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=True, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_xabft    = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=False, seed1=seed1, seed2=seed2, fault=True)
    network_prop     = MNN(size=MNN_array, activation_function=activation_function, batch_norm=batch_norm, mu=mu, std=std, row_test_size=row_test_size, col_test_size=col_test_size, soft_fault_rate=soft_fault_rate, hard_fault=True, baseline=False, prop=True, seed1=seed1, seed2=seed2, fault=True)

    batch_size = mini_batch_size
    batch_loop = int(np.ceil((100 / mini_batch_size)))

else:
    print("dataset名が不明です．use_datasetを確認してください．")
    sys.exit()

# ハイパラlamについて
# Adamの場合，lasso lam = 0.0005, glasso lam = 0.0018でした．

if optimizer_name == "SGD":
    optimizer = SGD()
elif optimizer_name == "AdaGrad":
    optimizer = AdaGrad()
elif optimizer_name == "Adam":
    optimizer = Adam()
else:
    print("optimizerが不明です．optimizer_nameを確認してください．")
    sys.exit()

epoch = 2000

train_size = x_train.shape[0]
test_size = x_test.shape[0]

# test_acc_list: 識別率(accuracy)を保存するリスト
test_acc_list_FF       = []
test_acc_list_baseline = []
test_acc_list_xabft    = []
test_acc_list_prop     = []

# test_error_list: 交差エントロピー誤差(error)を保存するリスト
test_error_list_FF       = []
test_error_list_baseline = []
test_error_list_xabft    = []
test_error_list_prop     = []

# 学習データ保存用ディレクトリ・フォルダの作成．
os.makedirs(str(use_dataset)+"/"+str(optimizer_name)+"/"+str(activation_function)+"/hidden"+str(hidden_name)+"/batchnormalization_"+str(batch_norm)+"/mu"+str(int(mu))+"-std"+str(int(std))+"/soft-fault-rate-"+str(int(soft_fault_rate * 100)), exist_ok = True)

# 識別率を保存するファイルの生成
with open(str(use_dataset)+"/"+str(optimizer_name)+"/"+str(activation_function)+"/hidden"+str(hidden_name)+"/batchnormalization_"+str(batch_norm)+"/mu"+str(int(mu))+"-std"+str(int(std))+"/soft-fault-rate-"+str(int(soft_fault_rate*100))+"/accuracy-row"+str(int(row_test_size))+"-col"+str(int(col_test_size))+".dat", mode="w") as f:
    f.write("epoch, fault-free, X-ABFT, propose")
    f.write("\n")

# 交差エントロピー誤差を保存するファイルの生成
with open(str(use_dataset)+"/"+str(optimizer_name)+"/"+str(activation_function)+"/hidden"+str(hidden_name)+"/batchnormalization_"+str(batch_norm)+"/mu"+str(int(mu))+"-std"+str(int(std))+"/soft-fault-rate-"+str(int(soft_fault_rate*100))+"/error-row"+str(int(row_test_size))+"-col"+str(int(col_test_size))+".dat", mode="w") as f:
    f.write("epoch, fault-free, X-ABFT, propose")
    f.write("\n")

elapsed_time = 0

# 学習開始．最初に決めたepoch回だけ繰り返す．
for q in range(epoch):
    print(str(q+1)+"loop")
    start = time.time()
    for k in range(batch_loop):
        # print(str(k+1)+"batch loop")
        np.random.seed(q*batch_loop + k)
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # crossbarの重みをセットする．構成上入れるのがやむを得ず．
        network_xabft.crossbar_weight_set()
        network_prop.crossbar_weight_set()

        # soft fault発生
        network_xabft.soft_fault_apply()
        network_prop.soft_fault_apply()
        # network_baseline.soft_fault_apply()

        # hard fault発生
        network_xabft.hard_fault_apply()
        network_prop.hard_fault_apply()
        # network_baseline.hard_fault_apply()

        # soft と hard を合体
        network_xabft.combine_fault()
        network_prop.combine_fault()
        # network_baseline.combine_fault()

        if (test_epoch % int(tau) == 0):
            # testして故障位置を特定する
            network_xabft.test_apply()
            network_prop.test_apply()

            # 故障の修復
            network_xabft.correction_apply()
            network_prop.correction_apply()

            # あとで詳細確認
            # network_xabft.set_Affine()
            # network_prop.set_Affine()

        # 勾配の算出
        grads_FF       = network_FF.gradient(x_batch, t_batch)
        # grads_baseline = network_baseline.gradient(x_batch, t_batch)
        grads_xabft    = network_xabft.gradient(x_batch, t_batch)
        grads_prop     = network_prop.gradient(x_batch, t_batch)

        # 重みの更新
        optimizer.update(network_FF.params, grads_FF)
        # optimizer.update(network_baseline.params, grads_baseline)
        optimizer.update(network_xabft.params, grads_xabft)
        optimizer.update(network_prop.params, grads_prop)

        # Affine_MEMの方で保存されている重みの更新
        network_FF.affine_update()
        # network_baseline.affine_update()
        network_xabft.affine_update()
        network_prop.affine_update()

        test_epoch += 1

    acc_FF, error_FF             = network_FF.accuracy(x_test, t_test)
    #acc_baseline, error_baseline = network_baseline.accuracy(x_test, t_test)
    acc_xabft, error_xabft       = network_xabft.accuracy(x_test, t_test)
    acc_prop, error_prop         = network_prop.accuracy(x_test, t_test)

    acc_FF       = np.around(acc_FF, 4)
    #acc_baseline = np.around(acc_baseline, 4)
    acc_xabft    = np.around(acc_xabft, 4)
    acc_prop     = np.around(acc_prop, 4)

    error_FF       = np.around(error_FF, 4)
    #error_baseline = np.around(error_baseline, 4)
    error_xabft    = np.around(error_xabft, 4)
    error_prop     = np.around(error_prop, 4)

    test_acc_list_FF.append(np.array(acc_FF).tolist())
    #test_acc_list_baseline.append(np.array(acc_baseline).tolist())
    test_acc_list_xabft.append(np.array(acc_xabft).tolist())
    test_acc_list_prop.append(np.array(acc_prop).tolist())

    test_error_list_FF.append(np.array(error_FF).tolist())
    #test_error_list_baseline.append(np.array(error_baseline).tolist())
    test_error_list_xabft.append(np.array(error_xabft).tolist())
    test_error_list_prop.append(np.array(error_prop).tolist())

    print("accuracy(fault free) : " + str(test_acc_list_FF[q]))
    #print("accuracy(baseline) : " + str(test_acc_list_baseline[q]))
    print("accuracy(X-ABFT) : " + str(test_acc_list_xabft[q]))
    print("accuracy(propose) : " + str(test_acc_list_prop[q]))

    print("error(fault free) : " + str(test_error_list_FF[q]))
    #print("error(baseline) : " + str(test_error_list_baseline[q]))
    print("error(X-ABFT) : " + str(test_error_list_xabft[q]))
    print("error(propose) : " + str(test_error_list_prop[q]))

    # 1epoch毎に学習データ（accuracy）を保存
    with open(str(use_dataset)+"/"+str(optimizer_name)+"/"+str(activation_function)+"/hidden"+str(hidden_name)+"/batchnormalization_"+str(batch_norm)+"/mu"+str(int(mu))+"-std"+str(int(std))+"/soft-fault-rate-"+str(int(soft_fault_rate*100))+"/accuracy-row"+str(int(row_test_size))+"-col"+str(int(col_test_size))+".dat", mode="a") as f:
        f.write(str(q+1)+", ")
        f.write(str(test_acc_list_FF[q])+", ")
        #f.write(str(test_acc_list_baseline[q])+", ")
        f.write(str(test_acc_list_xabft[q])+", ")
        f.write(str(test_acc_list_prop[q]))
        f.write("\n")

    with open(str(use_dataset)+"/"+str(optimizer_name)+"/"+str(activation_function)+"/hidden"+str(hidden_name)+"/batchnormalization_"+str(batch_norm)+"/mu"+str(int(mu))+"-std"+str(int(std))+"/soft-fault-rate-"+str(int(soft_fault_rate*100))+"/error-row"+str(int(row_test_size))+"-col"+str(int(col_test_size))+".dat", mode="a") as f:
        f.write(str(q+1)+", ")
        f.write(str(test_error_list_FF[q])+", ")
        #f.write(str(test_error_list_baseline[q])+", ")
        f.write(str(test_error_list_xabft[q])+", ")
        f.write(str(test_error_list_prop[q]))
        f.write("\n")

    elapsed_time = time.time() - start
    print(str(q+1)+"epoch time is :"+str(elapsed_time))

