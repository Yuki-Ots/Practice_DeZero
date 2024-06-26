import dezero
import dezero.functions as F
import numpy as np

x = dezero.Variable(np.random.randn(2, 10))
y = x.sum(axis=0)
y.backward()
print(y)
print(x.grad)

x = dezero.Variable(np.random.randn(2, 10))
y = F.sum(x, axis=None, keepdims=True)
print(y)
