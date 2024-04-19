"""
4/19 メモ
Variableクラスを被せることで自動微分をやり易くしている。
具体的にはVariableクラスに値、値を生成した関数、勾配を保存している
Variableクラスのメンバ関数であるbackwardで、保存した関数のbackward()を呼び出す。

TODO
なぜ funcというリストを作るの？
assert type(self.data) == np.ndarray 使い方が間違っているようだ。来週確認する。

"""

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported (a numpy array is supported)'.format(type(data)))

        self.data = data
        # assert type(self.data) == np.ndarray 使い方が間違っているようだ。来週確認する。
        self.grad = None
        # assert type(self.data) == np.ndarray
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]  # なぜ funcというリストを作るの？
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


# (数学の)関数を実装するの基底クラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 具体的な計算はforwardをオーバーライドして書く
        output = Variable(as_array(y))
        output.set_creator(self)  # 出力変数に産みの親を覚えさせる
        self.input = input  # 入力した値を覚えておく
        self.output = output  # 出力も覚える
        return output

    def forward(self, input: Variable):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0)
        return y

    def backward(self, gy):
        x = self.input.data
        if x >= 0:
            return gy
        else:
            return np.zeros_like(gy)

class Step(Function):
    def forward(self, x):
        y = (x >= 0)
        return y

    def backward(self, gy):
        return np.zeros_like(gy)


def numerical_diff(f, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def relu(x):
    f = ReLU()
    return f(x)


def step(x):
    f = Step()
    return f(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

