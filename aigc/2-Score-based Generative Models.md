# Score-based Generative Models

- 内容来自于大一统模型，大一统模型笔记太卡了，但score-based模型比较重要，所以单独写。
- 不是song yang论文的笔记

---

三种等价形式中证明，变分扩散模型可以通过优化一个神经网络 $s_\theta (x_t, t)$ 来匹配score function $\nabla log(p(x_t))$ 。然而，上面的推导中直接从 Tweedie's Formula 中得到了 score function 的表达式，但是没有从直观上理解分数函数到底是什么，也不知道为什么可以建模分数函数。幸运的是，可以通过另一种 Score-based 生成模型来更直观的理解。我们可以证明变分扩散模型 （VDM）有一种等价的 Score-based Generative 建模形式。

为了理解为什么优化分数函数有意义，首先借助能量模型引入一种能够表示任意概率分布的形式：
$$
p_\theta (x) = \frac{1}{Z_\theta} e^{-f_\theta (x)}
$$
  其中，

- $f_\theta (x)$ 是任意灵活的，可参数化的函数。叫做 能量函数 （Energy function）。通常使用神经网络进行建模。
- $Z_\theta$ 是一个归一化常量，来保证 $\int p_\theta(x) dx = 1$ 。因此，$Z_\theta = \int e^{-f_\theta (x)}$ 。同时也可以发现，用神经网络，或用其他比较复杂的 $f_\theta (x)$ 时，$Z_\theta$ 十分难以计算。

为了计算 $Z_\theta$ ，有一种近似方法。如果用极大似然估计，则计算 $p_\theta (x) $ 的对数似然的极大值：
$$
\nabla_x log(p(x_t)) = \nabla log(\frac{1}{Z_\theta} e^{-f_\theta (x)})
\\=

\frac{Z_\theta}{e^{-f_\theta (x)}} \frac{1}{Z_\theta} e^{-f_\theta (x)} (-\nabla f_\theta(x))

\\=

-\nabla f_\theta (x)
$$
结合大一统笔记中的推导，让神经网络 $s_\theta (x_t, t)$ 近似估计 $\nabla_x log(p(x_t))$ ，即：
$$
s_\theta (x_t, t) \approx -\nabla f_\theta (x)
$$

---

$s_\theta (x_t, t)$ 模型可以与 gt 的 score function $\nabla_x log(p(x_t))$ 计算 `费舍散度（Fisher Divergence）` ：
$$
E_{p(x)} [|| s_\theta(x) - \nabla logp(x) ||_2^2]
$$

- 费舍散度的定义：对于两个概率函数 $p(x), q(x)$ ，费舍散度的计算方式为 :
  $$
  E_{p(x)} || \nabla_x log(p(x)) - \nabla_x log(q(x)) ||_2^2
  $$

  其中，$p(x)$ 是真实的概率分布函数

---

从优化极大似然的梯度角度看， $\nabla logp(x)$



  

  

  