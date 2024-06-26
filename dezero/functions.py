import dezero
from dezero import Variable, Function, as_variable, no_grad
from dezero import utils
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


def tanh(x):
    return Tanh()(x)


class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        return gy / x

def log(x):
    return Log()(x)


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


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)



class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


# class BroadcastTo(Function):
#     def __init__(self, shape):
#         self.shape = shape
#
#     def forward(self, x):
#         self.x_shape = x.shape
#         xp = dezero.cuda.get_array_module(x)
#         y = xp.broadcast_to(x, self.shape)
#         return y
#
#     def backward(self, gy):
#         gx = sum_to(gy, self.x_shape)
#         return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return np.sum(diff ** 2) / len(diff)

    def backward(self, gy):
        x0, x1 = self.inputs
        gy = broadcast_to(gy, x0.shape)
        gx0 = gy * (x0 - x1) * (2.0 / len(x0))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x, W, b):
        if b is None:
            y = np.matmul(x, W)
        else:
            y = np.matmul(x, W) + b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        gb = None if b.data is None else sum_to(gy, b.shape)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x):
        y = 1. / (1. + np.exp(-x))
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1. - y)


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0.0
        return y

    def backward(self, gy):
        gx = 0.0 + gy
        gx.data[self.mask] = 0.0
        return gx

def relu(x):
    return ReLU()(x)


def numerical_grad(f, x:Variable, eps=1e-6):
    h = eps
    it = np.nditer(x.data, flags=['multi_index'])

    with no_grad():
        x = Variable(x.data.astype(np.float64))
        y = f(x)
        grads = np.zeros((*x.shape, *y.shape))
        while not it.finished:
            idx = it.multi_index
            tmp = x.data[idx]
            x.data[idx] = tmp + h
            fx2 = f(x)
            x.data[idx] = tmp - h
            fx1 = f(x)
            grads[idx] = (fx2.data - fx1.data) / (2 * h)
            x.data[idx] = tmp
            it.iternext()
    return grads
