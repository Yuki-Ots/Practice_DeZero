import dezero
import dezero.functions as F
import dezero.utils as utils
import numpy as np

x = dezero.Variable(np.array([1.0, 2.0, 3.0]), name='x')

y = F.sin(x); y.name = 'y'
z = F.sin(y); z.name = 'z'

z.backward(retain_grad=True, create_graph=True)

y.grad.name = 'y.grad'
z.grad.name = 'z.grad'
x.grad.name = 'x.grad'

utils.plot_dot_graph(x.grad, verbose=True, to_file='~/program/pythonPgms/aho3.png')
