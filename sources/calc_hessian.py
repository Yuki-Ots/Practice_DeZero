import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
import matplotlib.pyplot as plt

# None になる場合があるのでどうするか x** 2 + y ** 2などはgx = 2 * xだから yの勾配がNone -> 0を代入すればいいのでは？


def hoge(x, y):
    return x ** 2 + y ** 2


def rosenbrock(x, y):
    z = 100 * (y - x ** 2) ** 2 + (x - 1) ** 2
    return z


def rastrigin(x, y):
    A = 10
    z = A * 2 + (x ** 2 - 10 * A * F.cos(2 * np.pi * x)) + (y ** 2 - 10 * A * F.cos(2 * np.pi * y))
    return z


def plot3d(x, y):
    pass

def quadro_optimize(func, init_x, init_y, eps=1e-7):
    x = as_variable(init_x)
    y = as_variable(init_y)
    hesse = np.zeros((2, 2))
    d = 10
    i = 0
    while d > eps:
        z = func(x, y)
        z.backward(create_graph=True)
        rdx = x.grad
        rdy = y.grad

        x.cleargrad()
        y.cleargrad()

        rdx.backward()
        rdxgx = x.grad.data if x.grad is not None else 0
        rdygx = y.grad.data if y.grad is not None else 0
        hesse[:, 0] = np.array([rdxgx, rdygx])

        x.cleargrad()
        y.cleargrad()
        rdy.backward()
        rdxgy = x.grad.data if x.grad is not None else 0
        rdygy = y.grad.data if y.grad is not None else 0
        hesse[:, 1] = np.array([rdxgy, rdygy])

        delta = - np.linalg.inv(hesse) @ np.array([rdx.data, rdy.data])
        x.data += delta[0]
        y.data += delta[1]
        d = (delta ** 2).sum()
        print(f'{i} th {x, y}')
        i += 1



x = Variable(np.array(2.0))
y = Variable(np.array(1.0))

quadro_optimize(rastrigin, x, y)

n = 15
trace = np.zeros((n, 2))

hesse = np.zeros((2, 2))

z = hoge(x, y)


# for i in range(n):
#     trace[i] = np.array([x.data, y.data])
#     z = rosenbrock(x, y)
#     z.backward(create_graph=True)
#     rdx = x.grad
#     rdy = y.grad
#
#     x.cleargrad()
#     y.cleargrad()
#
#     rdx.backward()
#     rdxgx = x.grad.data if x.grad is not None else 0
#     rdygx = y.grad.data if y.grad is not None else 0
#     hesse[:, 0] = np.array([rdxgx, rdygx])
#
#     x.cleargrad()
#     y.cleargrad()
#     rdy.backward()
#     rdxgy = x.grad.data if x.grad is not None else 0
#     rdygy = y.grad.data if y.grad is not None else 0
#     hesse[:, 1] = np.array([rdxgy, rdygy])
#
#     print(hesse)
#
#     delta = - np.linalg.inv(hesse) @ np.array([rdx.data, rdy.data])
#     x.data += delta[0]
#     y.data += delta[1]
#     print(x, y)
#

# nanになるなんで？
# /Users/ootsukayuuki/PycharmProjects/PracticeDFlow/.venv/bin/python /Users/ootsukayuuki/PycharmProjects/PracticeDFlow/sources/calc_hessian.py
# [[2. 0.]
#  [0. 2.]]
# variable(0.0) variable(0.0)
# [[ 2.  0.]
#  [ 0. nan]]
# variable(nan) variable(nan)
# [[2. 0.]
#  [0. 2.]]
# variable(nan) variable(nan)
