# Estimation of non-normalized statistical models by score matching

# 0 引用

论文：https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf



# 1 介绍

大多数情况下，概率模型在机器学习，统计学，信号处理领域都是一种非归一化（non-normalized）概率密度。这意味着，这些模型还包含着一个未知的归一化常量，而归一化常量的计算通常是困难的。

假设我们观测到一个随机向量 $x \in \mathbb{R}^n$ ，其概率密度函数 (PDF) 定义为 $p_x(\cdot)$ 。有一个参数化的概率密度模型 $p(\cdot; \theta)$ ，$\theta$ 是一个 $m$ 维的参数向量。我们想从数据 $x$ 中估计参数 $\theta$  ，比如，我们想通过估计 $p(\cdot; \theta)$ 来近似 $p_x(\cdot)$ 。

---

用能量函数表示概率密度函数为：
$$
p(\xi;\theta) = \frac{1}{Z(\theta)} q(\xi; \theta)
$$

- 分子称为能量函数
- 分母称为归一化常量

为了使得上述概率密度函数的形式满足积分为1的条件：
$$
\int_\xi p(\xi;\theta) d\xi = \int_\xi \frac{1}{Z(\theta)} q(\xi; \theta) d\xi 
\\=

\frac{1}{Z(\theta)} \int_\xi  q(\xi; \theta) d\xi = 1
$$
因此：
$$
Z(\theta) = \int_\xi  q(\xi; \theta) d\xi
$$
在维度较高（实际上大多数情况下维度大于2时），$Z(\theta)$ 的数值解计算几乎就是不可能的了。

# 2 Score Matching

原论文的符号稍微有些奇怪，后面的推导都用song和常见的符号。定义：

- 概率密度函数  = $\frac{1}{Z(\theta)} e^{f_\theta(x)}$
- $Z(\theta) = \int_x  e^{f_\theta(x)} dx$ 
- 未归一化的概率密度函数 $q(x;\theta)$
- 分数函数（score function） $s(x; \theta)$

由于计算 $Z_\theta$ 需要计算上面那个比较困难的积分，所以Score Matching的核心思想就是，不直接去计算 $\frac{1}{Z(\theta)} e^{f_\theta(x)}$ ，而是尝试去看看 $\frac{1}{Z(\theta)} e^{f_\theta(x)}$ 也就是最终的目标  $p_\theta(x)$  的梯度能否拿来用一用。

- 直接计算梯度为：
  $$
  \nabla_x p_\theta(x) = \nabla_x \frac{1}{Z(\theta)} e^{f_\theta(x)}
  \\=
  \frac{1}{Z(\theta)} \nabla_x  e^{f_\theta(x)}
  $$
  还是有 $Z_\theta$ 。

- 因此，Score Matching计算的是 $p_\theta(x)$ 的对数梯度：
  $$
  \nabla_x log(p_\theta(x)) = \nabla_x log(\frac{1}{Z(\theta)} e^{f_\theta(x)})
  \\=
  \nabla_x [log(\frac{1}{Z(\theta)}) + log(e^{f_\theta(x)})]
  \\=
  \nabla_x [- log(Z(\theta)) + f_\theta(x)]
  \\=
  \nabla_x f_\theta(x) - \nabla_xlog(Z(\theta))
  \\=
  \nabla_x f_\theta(x)
  $$
  计算对数梯度就巧妙的把与 $x$ 无关的 $Z_\theta$ 给消掉了。

---

回到原文中的定义：


$$
s(x;\theta)= \nabla_x log(p_\theta(x))
$$

- 其中，$s(x;\theta)$ 被称为 score function

- 如刚才推导，等式右侧最终与 $Z_\theta$ 无关，因此：
  $$
  s(x;\theta)= \nabla_x f_\theta(x)
  $$
  这里和论文里的形式其实是一样的，即论文中的 $q(x;\theta) = e^{f_\theta(x)}$ ，加上log都一样。

---

因此，现在的目标就是建模 score function $s(x;\theta)$ ，令其与观测似然 $log(p_\theta(x))$ 一致即可，从而引出最小二乘法的目标函数：
$$
J(\theta) = \frac{1}{2} \int_x p(x) || s(x;\theta) -  \nabla_x log(p_\theta(x))||^2 dx
\\=
\frac{1}{2} \mathbb{E}_{x} [|| s(x;\theta) -  \nabla_x log(p_\theta(x))||^2]
$$
最优化目标为：
$$
\theta^* = arg min_\theta J(\theta)
$$
然而，目标函数中 $log(p_\theta(x))$ 仍然是未知的。为了解决该问题，论文使用了几个定理。

## 2.1 Theorem 1

**定理1：** 假设分数函数 $s(x;\theta)$ 是可微的，则在一些弱规则条件下，目标函数可以表示为：
$$
J(\theta) = \int_x p(x) \sum_{i=1}^n [\partial_i s_i(x;\theta) + \frac{1}{2} s_i(x;\theta)^2] dx + C
$$
其中，$C$ 是个常数，不依赖于 $\theta$ 。$s_i, x_i$ 表示对 $x$ 的第 $i$ 个维度求偏导，$x$ 共有 $n$ 个维度。
$$
s_i(x;\theta) = \frac{\partial log(p_\theta(x))}{\partial x_i}
$$

$$
\partial_i s_i(x;\theta) = \frac{\partial s_i(x;\theta)}{\partial x_i} = \frac{\partial^2 log(p_\theta(x))}{\partial^2 x_i}
$$

---

证明如下。

- 把 $J(\theta) = \frac{1}{2} \int_x p(x) || s(x;\theta) -  \nabla_x log(p_\theta(x))||^2 dx$ 的范数打开 ：
  $$
  J(\theta) = 
  \frac{1}{2} \int_x p(x) || s(x;\theta) -  \nabla_x log(p_\theta(x))||^2 dx
  \\=
  \frac{1}{2} \int_x p(x) [|| s(x;\theta)||^2 + ||\nabla_xlog(p_\theta(x))||^2 -  2( \nabla_xlog(p_\theta(x))^T s(x;\theta) )] dx
  
  \\=
  \frac{1}{2}  \int_x p(x) || s(x;\theta)||^2 dx
  \\ +
  \frac{1}{2}  \int_x p(x) ||\nabla_xlog(p_\theta(x))||^2 dx
  \\
  -\int_x p(x) ( \nabla_xlog(p_\theta(x))^T s(x;\theta) ) dx
  $$

  - 第二项 $\frac{1}{2}  \int_x p(x) ||\nabla_xlog(p_\theta(x))||^2 dx$ 与参数 $\theta$ 无关，可看作常数项。
  
  - 第一项可以展开：
    $$
    \frac{1}{2}  \int_x p(x) || s(x;\theta)||^2 dx
    =
    \frac{1}{2}  \int_x p(x) \sum_{i=1}^n (s_i(x;\theta))^2 dx
    $$
    
  - 第三项：
    $$
    -\int_x p_\theta(x) \sum_{i=1}^{n} \frac{\partial log(p_\theta(x))}{\partial x_i} s_i(x;\theta) dx
    \\ =
    -\sum_{i=1}^{n} \int_x p_\theta(x)  \frac{\partial log(p_\theta(x))}{\partial x_i} s_i(x;\theta) dx
    \\=
    -\sum_{i=1}^{n} \int_x p_\theta(x) \frac{1}{p_\theta(x)} \frac{\partial p_\theta(x)}{\partial x_i} s_i(x;\theta) dx
    
    \\=
    -\sum_{i=1}^{n} \int_x  \frac{\partial p_\theta(x)}{\partial x_i} s_i(x;\theta) dx
    $$

为了继续化简第三项：

- 根据分步积分：

  - 因为 $(uv)' = u'v + uv'$
  - 所以 $\int (uv)' dx = f(u'v + uv')dx = \int u'v dx + \int uv' dx$
  - 所以 $\int u'v dx = \int(uv)'dx - \int uv' dx$ 
  - 又因为 $\int(uv)'dx = uv$ 
  - 所以 $\int u'v dx = \int(uv)'dx - \int uv' dx = uv - \int uv' dx$

- 上述第三项中的 $\int_x  \frac{\partial p_\theta(x)}{\partial x_i} s_i(x;\theta) dx$ 替换成：
  $$
  \int_x  \frac{\partial p_\theta(x)}{\partial x_i} s_i(x;\theta) dx = \int_x  u' v dx =  uv - \int uv' dx
  \\=
  p_\theta(x) s_i(x;\theta) |_{-\infty}^{+ \infty} - \int_x p_\theta(x) \frac{\partial s_i(x;\theta)}{\partial x_i} dx
  $$
  
- 由于 $p_\theta(x)$ 是概率分布，一般数据分布的概率密度会集中在某些值，如高斯分布集中在均值，或图像数据分布只集中在某些区域。因此当 $x$ 取值比较极端时，$p_\theta(x)$ 就会趋近于0。因此：
  $$
  lim_{x \to \infty} p_\theta(x) s_i(x;\theta) |_{-\infty}^{+ \infty} = 0
  $$

最后整理一下 $J(\theta)$ ：
$$
J(\theta) =
\frac{1}{2}  \int_x p_\theta(x) || s(x;\theta)||^2 dx
\\ +
\frac{1}{2}  \int_x p_\theta(x) ||\nabla_xlog(p_\theta(x))||^2 dx
\\
-\int_x p_\theta(x) ( \nabla_xlog(p_\theta(x))^T s(x;\theta) ) dx

\\=
\frac{1}{2}  \int_x p(x) \sum_{i=1}^n (s_i(x;\theta))^2 dx
\\+
C
\\
-\sum_{i=1}^{n} - \int_x p_\theta(x) \frac{\partial s_i(x;\theta)}{\partial x_i} dx

\\=
\frac{1}{2}  \int_x p_\theta(x) \sum_{i=1}^n (s_i(x;\theta))^2 dx
\\+
C
\\ +
\sum_{i=1}^{n} \int_x p_\theta(x) \frac{\partial s_i(x;\theta)}{\partial x_i} dx

\\=
\frac{1}{2}  \int_x p_\theta(x) \sum_{i=1}^n (s_i(x;\theta))^2 dx
+
\sum_{i=1}^{n} \int_x p_\theta(x) \frac{\partial s_i(x;\theta)}{\partial x_i} dx 
+ 
C

\\=
\frac{1}{2}  \int_x p_\theta(x) \sum_{i=1}^n (s_i(x;\theta))^2 dx
+
\int_x p_\theta(x) \sum_{i=1}^{n}\frac{\partial s_i(x;\theta)}{\partial x_i} dx 
+ 
C

\\=
\int_x p_\theta(x) \sum_{i=1}^n [\frac{1}{2}s_i(x;\theta)^2 + \frac{\partial s_i(x;\theta)}{\partial x_i}] dx
+
C

\\=
\int_x p_\theta(x) \sum_{i=1}^n [\frac{1}{2}s_i(x;\theta)^2 + \partial_i s_i(x;\theta)] dx
+
C

\\=

\mathbb{E}_x [\frac{1}{2}|| s_i(x;\theta) ||^2 + \nabla_x^2 s_i(x;\theta)  ||^2]

\\=

\mathbb{E}_x [\frac{1}{2}|| \nabla_x log(p_\theta(x)) ||^2 + \nabla_x^2 log(p_\theta(x))  ||^2]
$$

---

至此，就完整的推出了 score matching的目标函数：

- 不再需要观测似然 $log(p(x))$ 这个未知的概率密度函数了，自然也就不再需要 $p(\xi;\theta) = \frac{1}{Z(\theta)} q(\xi; \theta)$ 中的归一化常量了。
- 只需要观测似然的对数的一阶导和二阶导即可。

# 3 Example

需要注意，Score Matching 方法只是通过log和求导的方式巧妙的把归一化常数 $Z_\theta$ 消掉了，但是能量函数的 $f(x)$ / $q(x)$ 还是需要的。因为计算 $J(\theta)$ 时必须要用能量函数。

下面是论文中求解多元高斯概率密度函数的示例：

![image-20231218113926263](./imgs/44-Estimation of non-normalized statistical models by score matching/image-20231218113926263.png)

- 需要用到能量函数来计算 $s(\theta;x)$ 的一阶/二阶导。 
