import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.arange(6))
y = F.reshape(x, (2, 3))
y.backward(retain_grad=True)
print(f'y.grad\n{y.grad},\n x.grad\n{x.grad}')
