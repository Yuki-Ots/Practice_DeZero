from dezero import Variable
from dezero import Function
import numpy as np

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = np.exp(x) * gy
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


# class ReLU
