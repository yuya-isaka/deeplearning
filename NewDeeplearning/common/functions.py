# coding: utf-8
import numpy as np
import numba

def identity_function(x):
    return x

def step_function(x):
    return np.array(x > 0, dtype=np.int)

@numba.jit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

@numba.jit
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

@numba.jit
def tanh(x):
    tmp = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    tmp[np.isnan(tmp)] = 0
    return tmp

@numba.jit
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        # print("x.shape is : "+str(x.shape))
        # print("x type is : "+str(type(x)))
        x = np.array(x, dtype=np.float64)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


@numba.jit
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

@numba.jit
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    #print("kotti kitawa")
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    delta = 1e-10

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

@numba.jit
def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

