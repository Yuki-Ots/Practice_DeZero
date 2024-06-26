import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero.utils import plot_dot_graph

def f(x):
    y = F.exp(x) * F.sin(2 * x)
    return y


x = Variable(np.linspace(0, 5, 10000))
n = 6
fig, ax = plt.subplots(figsize=(8, 8))
plt.ylim(10000, -10000)
y = f(x)
legends = [r'$f^{(' + str(i) + r')}(x)$' for i in range(n)]

for i in range(n):
    x.cleargrad()
    y.backward(create_graph=True)
    ax.plot(x.data, y.data, label=legends[i])
    y = x.grad

ax.legend(loc='upper right')
plt.savefig('higher_order_derivative_3.png')
plt.show()

plot_dot_graph(y, to_file='higher_order_derivative_3_calc_graph.png')
