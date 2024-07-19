import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0), name='x')
y = f(x)
y.name = 'y'
y.backward(create_graph=True)
print(x.grad)

# 新しく x.gradの参照するインスタンスの参照を作る
gx = x.grad
x.cleargrad()
gx.name = 'x.grad'
gx.backward()
print(x.grad)
plot_dot_graph(y, to_file='test_2nd_diff.png')
