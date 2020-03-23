# coding: utf-8
import numpy as np
import numba

class SGD:

    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr

    @numba.jit
    def update(self, params, grads):
        for key in params.keys():
            # print("params[key] is "+str(params[key]))
            params[key] -= self.lr * grads[key]

    @numba.jit
    def update_f(self, params, grads, fault):
        for key in params.keys():
            params[key] -= self.lr * grads[key] * fault[key]

    @numba.jit
    def update_clip(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            params[key] = np.where(params[key] > 1, 1, params[key])
            params[key] = np.where(params[key] < -1, -1, params[key])

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] = self.m[key] + (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] = self.v[key] + (1 - self.beta2) * (grads[key]**2 - self.v[key])
 
            self.m[key] = np.array(self.m[key], dtype=np.float64)
            self.v[key] = np.array(self.v[key], dtype=np.float64)

            params[key] = params[key] - lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            params[key] = np.where(params[key] > 1, 1, params[key])
            params[key] = np.where(params[key] < -1, -1, params[key])

            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
