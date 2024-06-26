import numpy as np
from dezero import *
import dezero.functions as F
import dezero.utils as dzutils
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


x0 = Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 'x0')
x1 = Variable(np.array([8.0, -1.0, 2.0]), 'x1')

z = F.sigmoid(x0)
z.backward()
print(x0.grad)

utils.plot_dot_graph(z, to_file='~/Desktop/pic1.png')
