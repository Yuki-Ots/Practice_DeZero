## 疑問点　一覧
### なぜ１変数関数に対してもgxsをタプルにするのか
- for文で回して処理したいため。
-             if not isinstance(gxs, tuple):
                gxs = (gxs,)  # 次のfor文で回すためにイテラブルにする

### Step18.pyで微分が2倍になってしまう
- 問題: 誤差逆伝播を多変数関数に対応させたところ、求める微分の2倍の値を返してしまうようになってしまった。公開されているサンプルプログラム(Step18.py)でも同様に正しく動かない。
- 原因: 多変数関数においては引数の数だけ逆伝播の流れができるが、同一のVariableオブジェクトを複数回利用してしまうと、先に計算した流れの微分が残っていて加算されてしまうためにその分だけ大きい値を返してしまう。

検証
- main_sub2.py
- 実行結果 main_sub2_out.txt
```python
x = Variable(np.array(1.0))
a = square(x)
b = square(x)
y = add(a, b)
```
なら上手くいき
```python
x = Variable(np.array(1.0))
a = square(x)
b = square(x)
y = add(a, a)
```
だと上手くいかない。

上手くいかない例
- $y = 2x^2$の$x=1$における微分をmain_sub2.pyでは計算しているが結果はmain_sub2_out.txtの通り8.0と返してしまう。
