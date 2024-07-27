import unittest
import numpy as np
import dezero.functions as F
from dezero import Variable
import dezero.layers as L
class TestConv(unittest.TestCase):
    def test_conv_backward(self):
        n = 4
        x = np.array([[i for j in range(n)] for i in range(1, n + 1)])
        x = Variable(x.reshape(1, 1, *x.shape))
        conv = L.Conv2d(1, 3, 1, 0, True)
        y = conv(x)
        y.backward()

        # 勾配確認
        grads = F.numerical_grad(conv, x)
        grads_summed = np.sum(grads, axis=(-1, -2)).reshape(1, 1, 4, 4)
        self.assertTrue(np.allclose(x.grad.data, grads_summed))

    # out_channel = 1
    def test_conv_backward2(self):
        n = 28
        kernel_size = 3
        stride = 2
        pad = 1
        x = np.array([[i for j in range(n)] for i in range(1, n + 1)])
        x = Variable(x.reshape(1, 1, *x.shape))
        input_size = x.shape[-1]
        output_size = F.get_conv_outsize(input_size, kernel_size, stride, pad)
        conv = L.Conv2d(1, kernel_size, stride, pad)
        y = conv(x)
        y.backward()

        # 勾配確認
        grads = F.numerical_grad(conv, x)
        grads_summed = np.sum(grads, axis=(-1, -2)).reshape(1, 1, input_size, input_size)
        self.assertTrue(np.allclose(x.grad.data, grads_summed))


    # outchannelが1でない場合
    def test_conv_backward3(self):
        n = 28
        kernel_size = 3
        stride = 2
        pad = 1
        out_channels = 3
        x = np.array([[i for j in range(n)] for i in range(1, n + 1)])
        x = Variable(x.reshape(1, 1, *x.shape))
        input_size = x.shape[-1]
        output_size = F.get_conv_outsize(input_size, kernel_size, stride, pad)
        conv = L.Conv2d(out_channels, kernel_size, stride, pad)
        y = conv(x)
        y.backward()

        # 勾配確認
        grads = F.numerical_grad(conv, x)
        grads_summed = np.sum(grads, axis=(-1, -2, -3)).reshape(1, 1, input_size, input_size)
        self.assertTrue(np.allclose(x.grad.data, grads_summed))


    # バッチサイズ込み(1)
    def test_conv_backward4(self):
        n = 28
        kernel_size = 3
        stride = 2
        pad = 1
        out_channels = 3
        batch_size = 3
        x = np.random.randn(batch_size, 1, n, n)
        x = Variable(x)
        input_size = x.shape[-1]
        output_size = F.get_conv_outsize(input_size, kernel_size, stride, pad)
        conv = L.Conv2d(out_channels, kernel_size, stride, pad)
        y = conv(x)
        y.backward()

        # 勾配確認
        grads = F.numerical_grad(conv, x)
        print(f'{grads.shape=}')
        grads_summed = np.sum(grads, axis=(-1, -2, -3, -4)).reshape(batch_size, 1, input_size, input_size)
        self.assertTrue(np.allclose(x.grad.data, grads_summed))


    # 完全版テスト(1)
    def test_conv_backward5(self):
        n = 28
        kernel_size = 3
        stride = 2
        pad = 1
        out_channels = 3
        batch_size = 3
        in_channels = 4
        x = np.random.randn(batch_size, in_channels, n, n)
        x = Variable(x)
        input_size = x.shape[-1]
        output_size = F.get_conv_outsize(input_size, kernel_size, stride, pad)
        conv = L.Conv2d(out_channels, kernel_size, stride, pad)
        y = conv(x)
        y.backward()

        # 勾配確認
        grads = F.numerical_grad(conv, x)
        print(f'{grads.shape=}')
        grads_summed = np.sum(grads, axis=(-1, -2, -3, -4)).reshape(batch_size, in_channels, input_size, input_size)
        self.assertTrue(np.allclose(x.grad.data, grads_summed))

    def test_conv_backward6(self):
        n = 30
        kernel_size = 3
        stride = 1
        pad = 1
        out_channels = 1
        batch_size = 3
        in_channels = 4
        x = np.random.randn(batch_size, in_channels, n, n)
        x = Variable(x)
        input_size = x.shape[-1]
        output_size = F.get_conv_outsize(input_size, kernel_size, stride, pad)
        conv = L.Conv2d(out_channels, kernel_size, stride, pad)
        y = conv(x)
        y.backward()

        # 勾配確認
        grads = F.numerical_grad(conv, x)
        print(f'{grads.shape=}')
        grads_summed = np.sum(grads, axis=(-1, -2, -3, -4)).reshape(batch_size, in_channels, input_size, input_size)
        self.assertTrue(np.allclose(x.grad.data, grads_summed))
