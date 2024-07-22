import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F

x = Variable(np.array([[1., 2., 3.]]))
y = F.softmax_simple(x)
print(y)
y.backward(retain_grad=True)

print(x.grad)
print(y.grad)
