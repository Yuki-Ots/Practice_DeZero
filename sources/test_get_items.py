import dezero
from dezero import Variable, Function
import numpy as np

class GetItem2(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = Variable(np.zeros(x.data.shape))
        gx.data[self.slices] = gy.data
        return gx

def getitem2(x, item):
    return GetItem2(item)(x)

class HogeVariable(Variable):
    def __getitem__(self, item):
        return getitem2(self, item)


class MyIterator:
    def __init__(self, *numbers):
        self._numbers = numbers
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        if self._index >= len(self._numbers):
            raise StopIteration()
        value = self._numbers[self._index]
        self._index += 1
        return value

    def __repr__(self):
        return 'MyIterator({})'.format(self._numbers)
