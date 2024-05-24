# Newton法を試す。
# 自動微分で2回微分を求める。
import numpy as np
from matplotlib import pyplot as plt

from dezero import Variable


def f(x):
    y = x ** 4 - 4 * x ** 2
    return y


x = Variable(np.array(3.0))
iter_time = 10
x_trace = np.zeros(iter_time)

for i in range(iter_time):
    x_trace[i] = x.data
    print(f'{i}th {x.data}')
    x.cleargrad()
    y = f(x)
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()  # 2階微分を入れる準備
    gx.backward()
    ggx = x.grad
    x.data -= gx.data / ggx.data


# x_trace2 = np.zeros(iter_time2)
#
# x_trace2[0] = x.data
# print(f'{0}th {x.data}')
# x.cleargrad()
# y = f(x)
# y.backward(create_graph=True)
# x -= lr * x.grad
#
# x_trace2[1] = x.data
# print(f'{1}th {x.data}')
# x.cleargrad()
# y = f(x)
# y.backward(create_graph=True)
# x -= lr * x.grad
#
#
lr = 0.01
iter_time2 = 150
x_trace2 = np.zeros(iter_time2)
x = Variable(np.array(3.0))
for i in range(iter_time2):
    x_trace2[i] = x.data
    print(f'{i}th {x.data}')
    x.cleargrad()
    y = f(x)
    y.backward(create_graph=True)
    x.data -= lr * x.grad.data

# for i in range(iter_time2):
#     x_trace2[i] = x.data
#     print(f'{i}th {x.data}')
#     y = f(x)
#     y.backward(create_graph=True)
#     gx = x.grad
#     x -= lr * gx

# X = np.linspace(-3.1, 3.1, 100)
# fig, ax = plt.subplots()
# # ax.plot(x_trace, f(x_trace), 'bo-', label='Newton method')
# # ax.plot(x_trace2, f(x_trace2), 'ro-', label='gradient decent method')
# ax.plot(X, f(X))
# ax.legend(loc='upper right')
# plt.savefig('gradient decent.png')
# plt.show()
