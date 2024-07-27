import numpy as np
import dezero
import math
import dezero.cuda

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)
# class Optimizer:
#     def __init__(self):
#         self.target = None
#         self.hooks = []
#
#     def setup(self, target):
#         self.target = target
#         return self
#
#     def update(self):
#         params = [p for p in self.target.params() if p.grad is not None]
#
#         # 前処理
#         for f in self.hooks:
#             f(params)
#
#         for param in params:
#             self.update_one(param)
#
#     def update_one(self, param):
#         raise NotImplementedError()
#
#     def add_hook(self, hook):
#         self.hooks.append(hook)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = dezero.cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros(param.data.shape)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = dezero.cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vs = {}
        self.ms = {}
        self.t = 0

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        key = id(param)
        xp = dezero.cuda.get_array_module(param.data)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        lr = self.alpha * xp.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        m = self.beta1 * self.ms[key] + (1 - self.beta1) * param.grad.data
        v = self.beta2 * self.vs[key] + (1 - self.beta2) * param.grad.data ** 2

        param.data -= lr * m / (xp.sqrt(v) + self.epsilon)
        self.ms[key], self.vs[key] = m, v


class NAG(Optimizer):
    """Nesterov's accelerated gradient descent optimizer."""
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.previous_data = {}
        self.previous_grads = {}

    def update_one(self, param):
        pass







class AdaGrad(Optimizer):
    pass
