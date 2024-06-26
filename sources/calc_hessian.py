import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt


x = Variable(np.linspace(-7, 7, 1000))
n = 4
fig, ax = plt.subplots(figsize=(8, 8))
y = F.sin(x)
legends = [r'$\sin^{(' + str(i) + r')}(x)$' for i in range(n)]

for i in range(n):
    x.cleargrad()
    y.backward(create_graph=True)
    ax.plot(x.data, y.data, label=legends[i])
    y = x.grad

ax.legend(loc='upper right')
plt.show()
