import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero
import dezero.utils as utils
import dezero.layers as L

n = 4
x = np.array([[i for j in range(n)] for i in range(1, n+1)])
x = Variable(x.reshape(1, 1, *x.shape))
conv = L.Conv2d(1, 3, 1, 0, True)
y = conv(x)
y.backward()
print(x.grad)

# 勾配確認
grads = F.numerical_grad(conv, x)
grads_summed = np.sum(grads, axis=(-1, -2)).reshape(1, 1, 4, 4)
print(grads_summed)

print(np.allclose(x.grad.data, grads_summed))

n = 4
x = np.array([[i for j in range(n)] for i in range(1, n+1)])
x = Variable(x.reshape(1, 1, *x.shape))
conv = L.Conv2d(1, 3, 1, 0, True)
y = conv(x)
y.backward()
print(x.grad)

# 勾配確認
grads = F.numerical_grad(conv, x)
grads_summed = np.sum(grads, axis=(-1, -2)).reshape(1, 1, 4, 4)
print(grads_summed)

print(np.allclose(x.grad.data, grads_summed))

#############################################################
# テスト結果
# variable([[[[ 0.22185174  0.30881286 -0.16131786 -0.24827898]
#             [ 0.59307703  0.8736621  -0.02007539 -0.30066045]
#             [ 0.73270947  1.03156433 -0.15753602 -0.45639088]
#             [ 0.36148417  0.46671508 -0.29877849 -0.4040094 ]]]])
# [[[[ 0.22185174  0.30881286 -0.16131786 -0.24827898]
#    [ 0.59307703  0.8736621  -0.02007539 -0.30066045]
#    [ 0.73270947  1.03156433 -0.15753602 -0.45639088]
#    [ 0.36148417  0.46671508 -0.29877849 -0.4040094 ]]]]
# True


















#
# n = 4
# x = np.array([[i for j in range(n)] for i in range(1, n+1)])
#
# # x = x.reshape(1, 1, *x.shape)
# # x
# # Out[8]:
# # array([[[[1, 1, 1, 1],
# #          [2, 2, 2, 2],
# #          [3, 3, 3, 3],
# #          [4, 4, 4, 4]]]])
# # x_col = F._im2col(x, 3, 2, 1, 0)
# # x_col
# # Out[10]:
# # variable([[1. 1. 2. 2.]
# #           [1. 1. 2. 2.]
# #           [1. 1. 2. 2.]
# #           [2. 2. 3. 3.]
# #           [2. 2. 3. 3.]
# #           [2. 2. 3. 3.]
# #           [3. 3. 4. 4.]
# #           [3. 3. 4. 4.]
# #           [3. 3. 4. 4.]])
# # dx_col = Variable(np.zeros(x_col.shape))
# # dx_col
# # Out[12]:
# # variable([[0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]
# #           [0. 0. 0. 0.]])
# # ls
# # README.md
# # __pycache__/
# # _dezero/
# # comprehension.md
# # dezero/
# # documents/
# # model.png
# # pictures/
# # sources/
# # test_import.py
# # tests/
# # dx_col = Variable(np.ones(x_col.shape))
# # dx_col
# # Out[15]:
# # variable([[1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]
# #           [1. 1. 1. 1.]])
# # dx = F._col2im(dx_col, x.shape, 3, 2, 1, 0)
# # dx
# # Out[17]:
# # variable([[[[1. 2. 2. 1.]
# #             [2. 4. 4. 2.]
# #             [2. 4. 4. 2.]
# #             [1. 2. 2. 1.]]]])
#
# /Users/ootsukayuuki/PycharmProjects/PracticeDFlow/.venv/bin/python /Users/ootsukayuuki/Applications/PyCharm Professional Edition.app/Contents/plugins/python/helpers/pydev/pydevconsole.py --mode=client --host=127.0.0.1 --port=59350
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/Users/ootsukayuuki/PycharmProjects/PracticeDFlow', '/Users/ootsukayuuki/PycharmProjects/PracticeDFlow/tests'])
# PyDev console: using IPython 8.24.0
# Python 3.12.0 (v3.12.0:0fb18b02c8, Oct  2 2023, 09:45:56) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
# exec(open("dezero/imports_list.py").read())
# n = 4
# x = np.array([[i for j in range(n)] for i in range(1, n+1)])
# x = x.reshape(1, 1, *x.shape)
# x_col = F._im2col(x, 3, 2, 1, 0)
# x_col
# Out[6]:
# variable([[1. 1. 2. 2.]
#           [1. 1. 2. 2.]
#           [1. 1. 2. 2.]
#           [2. 2. 3. 3.]
#           [2. 2. 3. 3.]
#           [2. 2. 3. 3.]
#           [3. 3. 4. 4.]
#           [3. 3. 4. 4.]
#           [3. 3. 4. 4.]])
# dx_col = Variable(np.zeros(x_col.shape))
# dx_col
# Out[8]:
# variable([[0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]
#           [0. 0. 0. 0.]])
# dx = F._col2im(dx_col, x.shape, 3, 2, 1, 0)
# dx
# Out[10]:
# variable([[[[0. 0. 0. 0.]
#             [0. 0. 0. 0.]
#             [0. 0. 0. 0.]
#             [0. 0. 0. 0.]]]])
# dx_col = Variable(np.ones(x_col.shape))
# dx = F._col2im(dx_col, x.shape, 3, 2, 1, 0)
# dx
# Out[13]:
# variable([[[[1. 2. 2. 1.]
#             [2. 4. 4. 2.]
#             [2. 4. 4. 2.]
#             [1. 2. 2. 1.]]]])
# x
# Out[14]:
# array([[[[1, 1, 1, 1],
#          [2, 2, 2, 2],
#          [3, 3, 3, 3],
#          [4, 4, 4, 4]]]])
# y = F._im2col(x, 3, 2, 1, 0)
# t
# Traceback (most recent call last):
#   File "/Users/ootsukayuuki/PycharmProjects/PracticeDFlow/.venv/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3577, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-16-34fc7a11cb38>", line 1, in <module>
#     t
# NameError: name 't' is not defined
# y
# Out[17]:
# variable([[1. 1. 2. 2.]
#           [1. 1. 2. 2.]
#           [1. 1. 2. 2.]
#           [2. 2. 3. 3.]
#           [2. 2. 3. 3.]
#           [2. 2. 3. 3.]
#           [3. 3. 4. 4.]
#           [3. 3. 4. 4.]
#           [3. 3. 4. 4.]])
# y.backward(retain_grad=True)
# x.grad
# Traceback (most recent call last):
#   File "/Users/ootsukayuuki/PycharmProjects/PracticeDFlow/.venv/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3577, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-19-82762e32cc47>", line 1, in <module>
#     x.grad
# AttributeError: 'numpy.ndarray' object has no attribute 'grad'
# x
# Out[20]:
# array([[[[1, 1, 1, 1],
#          [2, 2, 2, 2],
#          [3, 3, 3, 3],
#          [4, 4, 4, 4]]]])
# x = Variable(x)
# y = F._im2col(x, 3, 2, 1, 0)
# y.backward(retain_grad=True)
# x.grad
# Out[24]:
# variable([[[[1. 2. 2. 1.]
#             [2. 4. 4. 2.]
#             [2. 4. 4. 2.]
#             [1. 2. 2. 1.]]]])
# y.grad
# Out[25]:
# variable([[1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]
#           [1. 1. 1. 1.]])
# x.cleargrad()
# y = F._im2col(x, 3, 2, 1, 0)
# y
# Out[28]:
# variable([[1. 1. 2. 2.]
#           [1. 1. 2. 2.]
#           [1. 1. 2. 2.]
#           [2. 2. 3. 3.]
#           [2. 2. 3. 3.]
#           [2. 2. 3. 3.]
#           [3. 3. 4. 4.]
#           [3. 3. 4. 4.]
#           [3. 3. 4. 4.]])
# y.backward(retain_grad=True, create_graph=True)
# x.grad
# Out[30]:
# variable([[[[1. 2. 2. 1.]
#             [2. 4. 4. 2.]
#             [2. 4. 4. 2.]
#             [1. 2. 2. 1.]]]])
# gx = x.grad
# gx
# Out[32]:
# variable([[[[1. 2. 2. 1.]
#             [2. 4. 4. 2.]
#             [2. 4. 4. 2.]
#             [1. 2. 2. 1.]]]])
# x.cleargrad()
# gx.backward(retain_grad=True)
# x.grad
# gx.grad
# Out[36]:
# variable([[[[1. 1. 1. 1.]
#             [1. 1. 1. 1.]
#             [1. 1. 1. 1.]
#             [1. 1. 1. 1.]]]])
# x.grad
# utils.plot_dot_graph(gx, True, to_file='conv.png')
# Out[38]: <IPython.core.display.Image object>
# gx.backward()
# x.grad
# x is None
# Out[41]: False
# print(x)
# variable([[[[1 1 1 1]
#             [2 2 2 2]
#             [3 3 3 3]
#             [4 4 4 4]]]])
# x
# Out[43]:
# variable([[[[1 1 1 1]
#             [2 2 2 2]
#             [3 3 3 3]
#             [4 4 4 4]]]])
# x
# Out[44]:
# variable([[[[1 1 1 1]
#             [2 2 2 2]
#             [3 3 3 3]
#             [4 4 4 4]]]])
# x
# Out[45]:
# variable([[[[1 1 1 1]
#             [2 2 2 2]
#             [3 3 3 3]
#             [4 4 4 4]]]])
