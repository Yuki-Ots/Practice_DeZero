import dezero
from dezero import Variable, Function, as_variable, no_grad
from dezero import utils
import numpy as np
from dezero import cuda
from dezero.utils import pair


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sqrt(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sqrt(x)
        return y

    def backward(self, gy):
        x, = self.inputs[0]
        gx = gy / (2 * sqrt(x))
        return gx


def sqrt(x):
    return Sqrt()(x)


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = - gy * sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tan(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tan(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 + y ** 2)


def tan(x):
    return Tan()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        ex = xp.exp(x)
        emx = xp.exp(-x)
        y = (ex - emx) / (ex + emx)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)


def tanh(x):
    return Tanh()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
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
    if x.shape == shape:
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
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


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
        xp = cuda.get_array_module(x0)
        return xp.sum(diff ** 2) / len(diff)

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * (x0 - x1) * (2.0 / len(x0))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    # def forward(self, x, W, b):
    #     xp = cuda.get_array_module(x)
    #     if b is None:
    #         y = xp.matmul(x, W)
    #     else:
    #         y = xp.matmul(x, W) + b
    #     return y

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
        xp = cuda.get_array_module(x)
        y = 1. / (1. + xp.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1. - y)


def sigmoid(x):
    return Sigmoid()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = dezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


# class ReLU(Function):
#     def __init__(self):
#         self.mask = None
#     def forward(self, x):
#         self.mask = (x <= 0)
#         y = x.copy()
#         y[self.mask] = 0.0
#         return y
#
#     def backward(self, gy):
#         gx = (~self.mask).astype(np.float32) * gy
#         return gx

def relu(x):
    return ReLU()(x)


def numerical_grad(f, x: Variable, eps=1e-6):
    xp = cuda.get_array_module(x)
    h = eps
    it = xp.nditer(x.data, flags=['multi_index'])

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


# class Softmax(Function):
#     def __init__(self, axis):
#         self.axis = axis
#     def forward(self, x, axis=1):
#         x = x - np.max(x, axis=axis, keepdims=True)
#         exp_x = np.exp(x)
#         y = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
#         self.y = y
#         return y
#
#     def backward(self, gy):
#         y = self.outputs[0]()
#         y = - broadcast_to(y, (y.shape[1], y.shape[1]))
#         y += np.identity(y.shape[1])
#         y = self.y.T * y
#         return y

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=-1):
    return Softmax()(x)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlogp = log_p[np.arange(N), t.data]
    y = - 1 * sum(tlogp) / N
    return y


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


#
# class SoftmaxCrossEntropy(Function):
#     def __init__(self, axis=1, p_min=1e-15, p_max=1):
#         self.axis = axis
#         self.mask_clip = None
#         self.y = None
#         # self.p_min = p_min
#         # self.p_max = p_max
#     def forward(self, x, t):
#         axis = self.axis
#         xp = cuda.get_array_module(x)
#         N = x.shape[0]
#         x = x - xp.max(x, axis=axis, keepdims=True)
#         exp_x = xp.exp(x)
#         y = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
#         # self.mask_clip = (y >= self.p_min) * (y <= self.p_max)
#         # y = xp.clip(y, 1e-15, 1.0)
#         tlogy = xp.log(y[np.arange(N), t.ravel()])
#         loss = - xp.sum(tlogy) / np.float32(N)
#         return loss
#
#     def backward(self, gy):
#         x, t = self.inputs
#         N, COL = x.shape
#         gy *= 1 / N
#         xp = cuda.get_array_module(t.data)
#         eigen = xp.eye(COL, dtype=t.dtype)
#         t_one_hot = eigen[t.data]
#         y = softmax(x)
#         gx = gy * (y - t_one_hot)
#         return gx


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y_pred, target):
    y_pred, target = as_variable(y_pred), as_variable(target)
    y = y_pred.data.argmax(axis=-1).reshape(target.shape)
    result = (y == target.data)
    acc = result.mean()
    return Variable(np.array(acc))


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


def mean(x, axis=None, keepdims=False):
    return average(x, axis, keepdims)


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)
    if not dezero.Config.train:
        return x
    xp = cuda.get_array_module(x)
    mask = xp.random.rand(*x.shape) > dropout_ratio
    scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
    y = x * mask / scale
    return y


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


# 現状ではstrideやpadが異なる場合は対応していない。
def _im2col_array(x, fil_size, stride, pad):
    xp = cuda.get_array_module(x)
    x_b, x_c, x_h, x_w = x.shape
    fil_h, fil_w = pair(fil_size)
    stride, _ = pair(stride)
    pad, _ = pair(pad)
    y_h = get_conv_outsize(x_h, fil_h, stride, pad)
    y_w = get_conv_outsize(x_w, fil_w, stride, pad)
    index = -1

    x_pad = xp.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    x_col = xp.zeros((fil_h * fil_w, x_b, x_c, y_h, y_w))

    for h in range(fil_h):
        h2 = h + y_h * stride
        for w in range(fil_w):
            index += 1
            w2 = w + y_w * stride
            x_col[index, :, :, :, :] = x_pad[:, :, h:h2:stride, w:w2:stride]
    x_col = x_col.transpose(2, 0, 1, 3, 4).reshape(x_c * fil_h * fil_w, x_b * y_h * y_w)
    return x_col


def _col2im_array(dx_col, x_shape, fil_size, stride, pad):
    xp = cuda.get_array_module(dx_col)
    x_b, x_c, x_h, x_w = x_shape
    fil_h, fil_w = pair(fil_size)
    stride, _ = pair(stride)
    pad, _ = pair(pad)
    y_h = get_conv_outsize(x_h, fil_h, stride, pad)
    y_w = get_conv_outsize(x_w, fil_w, stride, pad)
    index = -1

    dx_col = dx_col.reshape(x_c, fil_h * fil_w, x_b, y_h, y_w).transpose(1, 2, 0, 3, 4)
    dx = xp.zeros((x_b, x_c, x_h + 2 * pad + stride - 1, x_w + 2 * pad + stride - 1))

    for h in range(fil_h):
        h2 = h + y_h * stride
        for w in range(fil_w):
            index += 1
            w2 = w + y_w * stride
            dx[:, :, h:h2:stride, w:w2:stride] += dx_col[index, :, :, :, :]

    return dx[:, :, pad:x_h + pad, pad:x_w + pad]


class _Im2col(Function):
    def __init__(self, fil_size, stride, pad):
        self.fil_size = pair(fil_size)
        self.y_size = None
        self.input_size = None
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x):
        self.input_size = x.shape
        y_h = get_conv_outsize(self.input_size[-2], self.fil_size[0], self.stride[0], self.pad[0])
        y_w = get_conv_outsize(self.input_size[-1], self.fil_size[1], self.stride[1], self.pad[1])
        self.y_size = (y_h, y_w)
        return _im2col_array(x, self.fil_size, self.stride, self.pad)

    def backward(self, gy):
        return _col2im(gy, self.input_size, self.fil_size, self.stride, self.pad)


def _im2col(x, fil_size, stride, pad):
    return _Im2col(fil_size, stride, pad)(x)


class _Col2im(Function):
    def __init__(self, input_size, fil_size, stride, pad):
        self.input_size = input_size
        self.fil_size = pair(fil_size)
        self.stride = pair(stride)
        self.pad = pair(pad)
        y_h = get_conv_outsize(self.input_size[-2], self.fil_size[0], self.stride[0], self.pad[0])
        y_w = get_conv_outsize(self.input_size[-1], self.fil_size[1], self.stride[1], self.pad[1])
        self.y_size = (y_h, y_w)

    def forward(self, dx_col):
        y = _col2im_array(dx_col, self.input_size, self.fil_size, self.stride, self.pad)
        return y

    def backward(self, gy):
        return _im2col(gy, self.fil_size, self.stride, self.pad)


def _col2im(x, input_size, kernel_size, stride, pad):
    return _Col2im(input_size, kernel_size, stride, pad)(x)


class Conv2d(Function):
    def __init__(self, stride, pad):
        self.stride = stride
        self.pad = pad
        self.input_shape = None
        self.output_shape = None
        self.kernel_shape = None

    def forward(self, x, W, b):

        xp = cuda.get_array_module(x)
        N, C, height, width = x.shape
        OC, _, KH, KW = W.shape
        self.input_shape = x.shape
        self.kernel_shape = W.shape
        self.W = W
        stride = self.stride
        pad = self.pad
        OH = get_conv_outsize(height, KW, stride, pad)
        OW = get_conv_outsize(width, KH, stride, pad)
        self.OH = OH
        self.OW = OW
        self.output_shape = N, OC, OH, OW
        self.x_col = _im2col_array(x, (KH, KW), self.stride, self.pad)
        self.w_col = W.reshape(OC, C * KH * KW)

        if b is None:
            y = xp.dot(self.w_col, self.x_col).T
        else:
            y = xp.dot(self.w_col, self.x_col).T + b
        self.y = y.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)

        return self.y

    def backward(self, gy):
        xp = cuda.get_array_module(gy)
        N, C, height, width = self.input_shape
        _, _, KH, KW = self.kernel_shape
        _, OC, OH, OW = self.output_shape
        dy = gy.transpose(0, 2, 3, 1).reshape(N * OH * OW, OC)
        dw = matmul(self.x_col, dy)

        self.dw = dw.T.reshape(OC, C, KH, KW)
        self.db = sum(dy, axis=0)

        dx_col = matmul(dy, self.w_col)
        self.dx = _col2im(dx_col.T, self.input_shape, (KH, KW), self.stride, self.pad)

        return self.dx


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = _im2col(x, (KH, KW), stride, pad)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y



def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)
    col = _im2col(x, kernel_size, stride, pad)

    col = col.T.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)
