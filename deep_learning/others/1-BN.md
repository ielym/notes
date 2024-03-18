# 1 BN 是不是线性的？

## 1.1 概念

线性映射指的是一个函数（function）或映射（map）。线性映射需要同时满足：

+ 可加性 ： $f(x + y) = f(x) + f(y)$
+ 齐次性： $f(ax) = a f(x)$

---

**通常我们认为 $f(x) = ax + b$ 是线性的，指的是其为一个线性方程，而不是线性映射/线性变换。实际上该函数是一个仿射变换，等于线性变换+平移** ：
$$
f(x + y) = a(x+y) + b = ax + ay + b \\
\ne f(x) + f(y) = (ax + b) + (ay + b) = ax + ay + 2b 
$$
**其中，如果只考虑 $ax$ 项，则显然是一个线性变换。**

---

**那么，卷积是不是线性变换呢？如果考虑 $f(x) = XW + B$ ，则其也是一个仿射变换。**

**问题是，可以把 $B$ 给合并到 $W$ 中，如：**
$$
\begin{bmatrix}
a_{00} & a_{01}  \\
a_{10} & a_{11} \\
a_{20} & a_{21} \\
\end{bmatrix}

\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix}

+ \begin{bmatrix}
b_{1}  \\
b_{2} \\
b_{3} \\
\end{bmatrix}
 = 
\begin{bmatrix}
a_{00} & a_{01} & b_{1}  \\
a_{10} & a_{11} & b_{2} \\
a_{20} & a_{21} & b_{3} \\

\end{bmatrix}

\begin{bmatrix}
x_1 \\
x_2 \\
1 \\
\end{bmatrix}
$$
**那么这是不是线性变换呢：**

+ **显然，上式是线性变换。只不过是对 $[X, 1]$ 的线性变换，而不是对 $X$ 的线性变换。**
+ **对 $X$ ，其还是一个仿射变换（线性变换+平移）**
+ **广义来看，一般说卷积是线性的，其实也就是将仿射变换看作是线性变换。当然，不带偏置的卷积一定是线性变换。**

---

## 1.2 BN 测试时

+ $\mu, \sigma, \gamma, \beta$ 都是常数，不随输入的变化而变化

+ BN 的函数为：
  $$
  f(x) = \gamma \frac{x - \mu}{\sigma}  + \beta = \frac{\gamma}{\sigma} x + (\beta - \frac{\gamma \mu}{\sigma})
  $$

+ 可以把 $(\beta - \frac{\gamma \mu}{\sigma})$ 看作是常数项

### 不考虑偏置项

+ 齐次性：
  $$
  f(ax) = \frac{\gamma}{\sigma} ax = af(x)
  $$

+ 可加性：
  $$
  f(x + y) = \frac{\gamma}{\sigma} (x+y) = \frac{\gamma}{\sigma}x + \frac{\gamma}{\sigma}y = f(x) + f(y)
  $$

因此，测试时如果不考虑偏置项，BN是线性的。

### 考虑偏置项

如果考虑偏置项，则BN实际上是仿射变换，但可以认为是线性变换。

## 1.3 BN训练时

+ $\mu, \sigma$ 随输入的变化而变化。$ \gamma, \beta$ 通过梯度下降法在变化，但是不随输入的变化而变化，暂且认为是两个常量。

### 不考虑偏置项

+ 齐次性：
  $$
  f(ax) = \frac{\gamma}{\sqrt{a^2\sigma^2}} ax = \frac{\gamma}{a\sigma} ax = \frac{\gamma}{\sigma} x \ne af(x)
  $$

+ 可加性（假设x, y 相互独立）：
  $$
  f(x + y) = \frac{\gamma}{\sqrt{\sigma_x^2 + \sigma_y^2}} (x+y) \ne f(x) + f(y)
  $$

因此，训练时即使不考虑偏置项，BN也是非线性的。

## 1.4 结论

+ 训练时，BN是仿射变换，但可以近似认为是线性变换。
+ 训练时，即使不考虑平移（偏置），BN也是非线性变换。