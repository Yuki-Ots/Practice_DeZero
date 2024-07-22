# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
import math


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

lr = 0.001
iters = 10000
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

x0_trace = np.zeros(iters)
x1_trace = np.zeros(iters)

for i in range(iters):
    x0_trace[i] = x0.data
    x1_trace[i] = x1.data

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad  # 勾配降下法
    x1.data -= lr * x1.grad
print(x0, x1)
fig = plt.figure(figsize=(4, 4))
x0min, x0max = -2, 2
x1min, x1max = -1, 3
nx, ny = 100, 100
x = np.linspace(x0min, x0max, nx+1)
y = np.linspace(x1min, x1max, ny+1)
XX, YY = np.meshgrid(x,y)
ZZ = rosenbrock(XX, YY)
fig, ax=plt.subplots()
ax.contour(XX,YY,ZZ, levels=30)
ax.plot(x0_trace, x1_trace, 'ro-')
plt.show()

# x = np.linspace(-1, 1, 100)
# y = np.linspace(-1, 1, 100)
# XX, YY = np.meshgrid(x, y)
# ZZ = rosenbrock(XX, YY)
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(XX, YY, ZZ, alpha=1.0)
# # ax.contour(XX, YY, ZZ)
# # ax.plot(x0.data, x1.data, y.data, marker="o",linestyle='None', color='red')
# # ax.quiver(x.data, y.data, 0, x.data - x.grad, y.data - y.grad, 0,arrow_length_ratio=0.3)
# ax.set_xlabel(r'$x0$')
# ax.set_ylabel(r'$x1$')
# ax.set_zlabel(r'$y$')
# plt.savefig('rosenbrock.png')
# plt.show()

print(x0.grad, x1.grad)
