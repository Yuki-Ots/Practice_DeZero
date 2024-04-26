"""
TODO
- numerical_diff が使えなくなった　修正せよ
- x.grad += gx としてはいけない理由を理解せよ
- pythonの機能の理解 see study.txt
- Funcion class の forward(self, *xs)では？
- Funcion class の backward(self, *gys)では？
- funcはキューじゃダメなの？


4/19 メモ
Variableクラスを被せることで自動微分をやり易くしている。
具体的にはVariableクラスに値、値を生成した関数、勾配を保存している
Variableクラスのメンバ関数であるbackwardで、保存した関数のbackward()を呼び出す。

4/25メモ
- gx は gradient of xの略
- xs は xの複数形だが x.data のリスト
- clear_grad において self.grad = 0 じゃなくて None

4/26
- クラスの関係についてのメモを作成した
x = Variable(np.array(2.0))
y = add(add(x, x), x)
y.backward()
print('first time', x.grad)
x.clear_grad()
y = add(x, x)
y.backward()
print('second time', x.grad)

# clear_gradなし
# first time 3.0
# second time 5.0
# あり
# first time 3.0
# second time 2.0

x = Variable(np.array(2.0))
a = exp(x)
y = add(a, a)
y.backward()
print(x.grad)
# 29.5562243957226


DONE
なぜ funcというリストを作るの？ stackを作るため
assert type(self.data) == np.ndarray 使い方が間違っているようだ。来週確認する。
間違いでは内容だがassert isinstance(self.grad, np.ndarray)が適切

"""

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported (a numpy array is supported)'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]  # なぜ funcというリストを作るの？
        while funcs:
            print(f'funcs={funcs}')
            f = funcs.pop()
            print(f'f={f}, f.inputs={f.inputs}, f.outputs={f}', sep='\n')
            print(f'funcs={funcs} (after popped)')
            gys = [output.grad for output in f.outputs]
            print(f'gys={gys}')
            gxs = f.backward(*gys)
            print(f'gxs={gxs}')
            if not isinstance(gxs, tuple):
                gxs = (gxs,)  # 次のfor文で回すためにイテラブルにする

            for x, gx in zip(f.inputs, gxs):
                print(f'x.grad={x.grad}, gx={gx}')
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                print(f'x.grad={x.grad}, gx={gx}')

                if x.creator is not None:
                    funcs.append(x.creator)


# (数学の)関数を実装するの基底クラス
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 具体的な計算はforwardをオーバーライドして書く
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs  # 入力した値を覚えておく
        self.outputs = outputs  # 出力も覚える
        return outputs if len(outputs) > 1 else outputs[0]  # リストの要素が1つの時は最初の要素を返す

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        print("backward at square")
        x = self.inputs[0].data
        gx = 2 * x * gy
        print(f"gx= {gx}")
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        print("backward exp")
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        print(f"gx= {gx}, gy={gy}, np.exp(x) = {np.exp(x)}")
        return gx


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        if x >= 0:
            return gy
        else:
            return np.zeros_like(gy)


class Step(Function):
    def forward(self, x):
        y = (x >= 0)
        return y

    def backward(self, gy):
        return np.zeros_like(gy)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        print("backward at add")
        print(f'gy={gy}')
        return gy, gy


class Multiply(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy


def numerical_diff(f, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def relu(x: Variable) -> Variable:
    f = ReLU()
    return f(x)


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


def multiply(x0: Variable, x1: Variable) -> Variable:
    return Multiply()(x0, x1)


def step(x: Variable) -> Variable:
    f = Step()
    return f(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


x = Variable(np.array(1.0))
a = square(x)
b = square(x)
y = add(a, a)
print(f'x.grad={x.grad}, a.grad={a.grad}, y.grad={y.grad}')
print('y.backward()')
y.backward()
print(f'x.grad={x.grad}, a.grad={a.grad}, y.grad={y.grad}')
# 29.5562243957226


# clear_gradなし
# first time 3.0
# second time 5.0
# あり
# first time 3.0
# second time 2.0

# func = lambda x: add(add(x, x), x)
# numerical_diff(func, x)
