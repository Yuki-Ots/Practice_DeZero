from dezero import cuda
from dezero.core import Parameter
import weakref
import dezero.functions as F
import numpy as np
from dezero.utils import pair

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            value = self.__dict__[name]
            if isinstance(value, Layer):
                yield from value.params()
            else:
                yield value

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, param_dict, parent_key=""):
        for name in self._params:
            value = self.__dict__[name]
            key = parent_key +'/' +  name if parent_key else name
            if isinstance(value, Layer):
                value._flatten_params(param_dict, key)
            else:
                param_dict[key] = value

    def save_weights(self, path):
        self.to_cpu()
        param_dict = {}
        self._flatten_params(param_dict)
        array_dict = {key: param.data for key, param in param_dict.items()}
    def load_weights(self, path):
        npz = np.load(path)
        param_dict = {}
        self._flatten_params(param_dict)
        for key, param in param_dict.items():
            param.data = npz[key]




class Linear(Layer):
    def __init__(self, out_size, no_bias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if no_bias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        self.nobias = nobias

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def _init_b(self, xp=np):
        self.b = Parameter(xp.zeros(self.out_channels), name='b')

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        if not self.nobias:
            self._init_b(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
