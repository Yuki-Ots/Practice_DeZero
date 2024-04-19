import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data  # numpy.ndarrayを想定
        self.grad = None  # numpy.ndarrayを想定
        self.creator = None

    def set_creator(self, func):
        self.creator = func


# (数学の)関数を実装するの基底クラス
class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)  # 具体的な計算はforwardをオーバーライドして書く
        output = Variable(y)
        output.set_creator(self) # 出力変数に海の親を覚えさせる
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


# x = Variable(np.array([10, 0.5]))
#
# f = Square()
# x = Variable(np.array(2.0))
# y_prime = numerical_diff(f, x, eps=1e-4)

x = Variable(np.array(1.0))

A = Square()
B = Exp()
C = Square()

a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)


class Tmp(Function):
    def forward(self, x):
        a = x ** 2
        b= np.exp(a)
        y = b ** 2
        return y

    def backward(self, gy):
        pass


f = Tmp()
print(numerical_diff(f, x, eps=1e-4))
print(x.grad)
