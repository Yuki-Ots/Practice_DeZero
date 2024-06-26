import dezero
from dezero import Variable, Function, as_variable, as_array
from dezero import as_variable
import numpy as np

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sqrt(Function):
    def forward(self, x):
        y = np.sqrt(x)
        return y
    def backward(self, gy):
        x, = self.inputs[0]
        gx = gy / (2 * np.sqrt(x))
        return gx


def sqrt(x):
    return Sqrt()(x)


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = - gy * sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tan(Function):
    def forward(self, x):
        y = np.tan(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 + y ** 2)


def tan(x):
    return Tan()(x)

class Tanh(Function):
    def forward(self, x):
        ex = np.exp(x)
        emx = np.exp(-x)
        y = (ex - emx) / (ex + emx)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)


class ReShape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape  # backwardの時のためにxの元のshapeを保持しておく。
        y = x.reshape(self.shape)
        return y


    def backward(self, gy):
        gx = reshape(gy, self.x_shape)
        return gx


def reshape(x, shape):
    if x.shape == shape:  # reshapeする必要がなかった場合、Variableであることだけ保証して返す。保証されてないとIDEが型推論する時に正常に動作しない？
        return as_variable(x)
    return ReShape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

def transpose(x, axes):
    return Transpose(None)(x, axes)


def numerical_grad(f, x:Variable):
    h = 1e-4
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])

    with no_grad():
        while not it.finished:
            idx = it.multi_index
            tmp = x.data[idx]
            x.data[idx] = tmp + h
            fx2 = f(x)
            x.data[idx] = tmp - h
            fx1 = f(x)
            grad[idx] = (fx2.data - fx1.data) / (2 * h)

    return grad







# class ReLU
