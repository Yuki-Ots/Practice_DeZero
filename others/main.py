"""
TODO
- numerical_diff が使えなくなった　修正せよ
- x.grad += gx としてはいけない理由を理解せよ
- pythonの機能の理解 see study.txt
- variableもdata = as_variable(data)とする？
- なんだったのこれ
y.backward()
Out[94]: (-2.0,)

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

5/10
そうか！Functionインスタンスはoutputを弱参照としてもっているんだ！これによって循環参照を解消して

5/17
__repr__ vs __str__
__rper__ : - インタラクティブシェルなどで
        - __str__が未実装の時にprintで
        - repr()に渡した時

- なぜcだけ arrayなの？
mul はx1 * gyと演算しているため
b  = Variable(np.array(3.0))
b  = Variable(np.array(2.0))
c  = Variable(np.array(1.0))
y = add(mul(a, b), c)
y.backward()
backward at add
gy=1.0
print(y)
variable(7.0)
a.grad
Out[35]: 2.0
b.grad
Out[36]: 3.0
c.grad
Out[37]: array(1.)

type(a.grad)
Out[44]: numpy.float64
type(c.grad)
Out[45]: numpy.ndarray

5/18
インテリセンスが効かなくなった
どうやら演算子オーバロードが読み込まれてないみたい
yがfloat型と解釈されている
x = Variable(np.array(1.0))
y = (x + 3.0) ** 2
y.backward()


DONE
なぜ funcというリストを作るの？ stackを作るため
assert type(self.data) == np.ndarray 使い方が間違っているようだ。来週確認する。
-> is が正しいもしくはisinstanceを使えばほぼ同じことができる。
間違いでは内容だがassert isinstance(self.grad, np.ndarray)が適切
- funcはキューじゃダメなの？　だめ
- Funcion class の forward(self, *xs)では？　そう
- Funcion class の backward(self, *gys)では？　そう

"""

import numpy as np
import weakref
import contextlib


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


class Variable:
    __array__priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:  # 末端のVariableでないなら
                for y in f.outputs:
                    y().grad = None  # outputのgradを解放


def as_variable(obj) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


# (数学の)関数を実装するの基底クラス
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 具体的な計算はforwardをオーバーライドして書く
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])  # 入力の世代の最大の物に合わせる
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs  # 入力した値を覚えておく
            self.outputs = [weakref.ref(output) for output in outputs]  # weakrefで保持しておくことで循環参照を作らない

        return outputs if len(outputs) > 1 else outputs[0]  # リストの要素が1つの時は最初の要素を返す

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *gys):
        raise NotImplementedError()


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
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
        return gy, gy


def add(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy


def mul(x0: Variable, x1) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy * x1
        gx1 = - gy * x0 / (x1 ** 2)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


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
    return Exp()(x)


def relu(x: Variable) -> Variable:
    f = ReLU()
    return f(x)


def step(x: Variable) -> Variable:
    return Step()(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow


def no_grad():
    return using_config('enable_backprop', False)


# x = Variable(np.array(1.0))
# a = square(x)
# b = square(x)
# y = add(a, b)
# print(f'x.grad={x.grad}, a.grad={a.grad}, y.grad={y.grad}')
# print('y.backward()')
# y.backward()
# print(f'x.grad={x.grad}, a.grad={a.grad}, y.grad={y.grad}')
# # 29.5562243957226
# with no_grad():
#     x = Variable(np.array(2.0))
#     y = square(x)
#     y.backward()
#     print(x.grad)

# clear_gradなし
# first time 3.0
# second time 5.0
# あり
# first time 3.0
# second time 2.0

# func = lambda x: add(add(x, x), x)
# numerical_diff(func, x)
# for i in range(200):
#     x = Variable(np.random.randn(10000))
#     y = square(square(x))


x = Variable(np.array([1, 2, 3, 4, 5]))
y = (x + 3.0) ** 2
y.backward()
print(x.grad)
