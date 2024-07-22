import numpy as np

import dezero

from dezero import Variable, no_grad
import dezero.layers as layers
import dezero.functions as F

x0 = Variable(np.array([1, 2, 3], dtype=np.float32))
x1 = Variable(np.array([1, 2, 3], dtype=np.float32))

def f(x):
    return x ** 2


iter_num = 20
lr = 0.1

for i in range(iter_num):
    y = f(x0)
    x0.cleargrad()
    y.backward()
    with no_grad():
        x0 -= lr * x0.grad
    print(x0)

# メモ
# torchとdezeroでインプレース演算子の挙動が異なる
#
# pytroch
# import torch
# k = torch.tensor(np.array(1.))
# id(k)
# Out[13]: 4962214736
# y = torch.tensor(np.array(3.0))
# id(y)
# Out[15]: 4962215536
# y -= k
# id(y)
# Out[17]: 4962215536
#
# dezero
# a = Variable(np.array(3.0))
# a = Variable(np.array(3.0), 'a')
# y = 2 * a
# y.name = 'y' + str(id(y))
# id(y)
# Out[8]: 4875696224
# y -= a
# id(y)
# Out[10]: 4875686336
