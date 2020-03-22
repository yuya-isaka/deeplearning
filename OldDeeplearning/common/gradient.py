import numpy as np  


#勾配を求める（数値微分は時間がかかるので、基本は誤差逆伝播法を使うのが効率的だが、確認のために使ったりする。）
def _numerical_gradient_1d(f, x): #一次元の時
    h = 1e-4
    grad = np.zeros_like(x)

    for idx, tmp_val in enumerate(x):
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def _numerical_gradient_2d(f, X): #二次元のとき
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
            
        return grad


def numerical_gradient(f, x): #多次元のとき
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val
        it.iternext()
        
    return grad


#勾配降下法
def gradient_descent(f, init_x, lr=0.01, step_num=100): #最初の入力xの値を変えることで、fの関数にとって最小になるパラメータを求める、 #ある方程式の勾配を降下させる
    x = init_x

    for _ in range(step_num):
        grad = numerical_gradient(f, x) #勾配を返してくる

        x -= lr * grad #その勾配x学習率（こいつは人の手で決めたりするからハイパーパラメータという、大きすぎても小さすぎてもだめ）ぶんを引き算する

    return x
