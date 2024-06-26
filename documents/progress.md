
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
- is が正しいもしくはisinstanceを使えばほぼ同じことができる。
間違いでは内容だがassert isinstance(self.grad, np.ndarray)が適切
- funcはキューじゃダメなの？　だめ
- Funcion class の forward(self, *xs)では？　そう
- Funcion class の backward(self, *gys)では？　そう


5/29
インテリセンスの問題をとりあえず解決
したこと
再インストール
新しいバージョンにアップデート
設定PYTHONコンソールのところで2箇所チェックを入れ直して再起動
これをしたら
Variableの補完はするようになったけど演算子オーバーロードとかの補完が効かない。
y = (x0 + x1) ** 2としてもx0 + x1をVariable型と見ない。
そこで
```py
Variable.__add__ = add
```
とかを
```py
class Variable:
    ...
    def add(self, other):
        return add(self, other)
```
などとしたらうまく行った。
前の形式の書き方だと実行はうまくいくけどPycharmのインスペクタがちゃんと読んでくれないみたい。


### 問題
- どうして逆伝播の際に世代を設けることと、funcsをソートしておく必要があるのか述べよ。
- ダメな例を示せ。
- 自動微分とは何か説明せよ。
- x = Variable(1.0)はエラーとなるか。またその理由を述べよ。
- 以下のコードで生成される計算グラフの画像は逆伝播に関するものだけである。その理由を述べよ。

```py
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0), name='x')
y = f(x)
y.name = 'y'
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.name = 'x.grad'
gx.backward()
print(x.grad)
plot_dot_graph(gx, to_file='test_2nd_diff.png')

```
- 前の方でとしなければならなかった。
```py
x.grad = x.grad + gx 
```
1.  これはなぜか。
2.  x.gradをVariable型に変更した後もこのように書く必要があるか？

ニュートン法
1. ニュートン法は f(x) =0の解を求める手法であるが、本で紹介されているニュートン法と結局は同じであることを説明せよ。
2. ニュートン法が 収束する　。。。

準ニュートン法


トポロジカルソート

トポロジカルソートで実装せよ。

5/24
self.grad: Variable = Noneとしたら
gradにも補完が効く様になった。


```py
x -= lr *x.grad
と書いてはいけない理由

numpyと違って新しいインスタンスを生成しているため。
id(x)
Out[24]: 5347905552
x -= lr *x.grad
id(x)
Out[26]: 5347915584

x.data -= lr * x.grad.dataとする必要がある。
```

class Exp
backward
np.exp(x) * gy
だめ
Variableをnp.expに入れられない

循環参照
Traceback (most recent call last):
  File "/Users/ootsukayuuki/program/pythonPgms/hoge1.py", line 1, in <module>
    from hoge2 import hoge2hello
  File "/Users/ootsukayuuki/program/pythonPgms/hoge2.py", line 1, in <module>
    from hoge1 import hoge1hello
  File "/Users/ootsukayuuki/program/pythonPgms/hoge1.py", line 1, in <module>
    from hoge2 import hoge2hello
ImportError: cannot import name 'hoge2hello' from partially initialized module 'hoge2' (most likely due to a circular import) (/Users/ootsukayuuki/program/pythonPgms/hoge2.py)
