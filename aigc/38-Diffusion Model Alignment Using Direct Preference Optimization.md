# Diffusion Model Alignment Using Direct Preference Optimization

![image-20231208113426315](/home/mi/.config/Typora/typora-user-images/image-20231208113426315.png)

# 0 引用

- 论文：https://arxiv.org/pdf/2311.12908.pdf
- DPO推导：https://zhuanlan.zhihu.com/p/634705904

# 1 介绍

- 不需要提前优化奖励函数

# 2 Background



## 2.1 奖励模型

使用 Bradley-Terry (BT) 模型，即猜胜率：
$$
p_{BT} (x_0^w > x_0^l | c) = \sigma (r(c, x_0^w) - r(c, x_0^l))
$$
其中：

- $r(c, x_0)$ 是一个神经网络模型，参数 $\phi$ ，对于条件 $c$ 和图像 $x_0$ ，输出一个类似分数的数值。
- $\sigma$ 是 Sigmoid函数，用于把两张图像的类分数的输出的差值转换成胜率大小。
- $p_{BT} (x_0^w > x_0^l | c)$ 表示在条件 $c$ 的条件下，$x_0^w$ （winning）胜过 $x_0^l$ （lossing）的胜率。

优化目标：

- 对于已经排序过的样本对 $x_0^w, x_0^l, c$ ，极大化观测对数似然来求解奖励打分模型的参数 $\phi$ ：

$$
L_{BT} (\phi) = - E_{c, x_0^w, x_0^l} [log(\sigma (r(c, x_0^w) - r(c, x_0^l)))]
$$

## 2.2 RLHF

RLHF的目的是优化一个模型 $p_\theta (x_0 | c)$ ，如最终想要获得的扩散模型，并使得 $p_\theta$ 在奖励模型上的分数 $r(c, x_0)$ 最大化。同时，强化学习时也需要对 $p_\theta (x_0 | c)$ 添加正则，防止输出与原模型 $p_{ref} (x_0 | c)$ 的差距过大。

- 第一个目标：给定条件 $c$ 和优化后的模型 $p_\theta (x_0 | c)$ ，采样出的样本 $x_0$ 在奖励模型上的分数需要最大化，即：
  $$
  max E_{c \in D_c, x_0 \in p_\theta(x_0 | c)} [r(c, x_0)]
  $$

- 第二个目标：优化后的模型 $p_\theta (x_0 | c)$ 不能和原模型 $p_{ref} (x_0 | c)$ 的输出差距太大 ，使用KL散度衡量差距，需要最大化负的KL散度，即：
  $$
  max -KL(p_\theta(x_0 | c) || p_{ref} (x_0 | c))
  $$

- 整体的强化学习目标是二者之和：
  $$
  max_{p_\theta} E_{c \in D_c, x_0 \in p_\theta(x_0 | c)} [r(c, x_0)] - \beta KL(p_\theta(x_0 | c) || p_{ref} (x_0 | c))
  $$
  其中，$\beta$ 是控制正则化项的系数

## 2.3 DPO Objective

从 RLHF 中的目标函数可以进一步推导：
$$
max_{p_\theta} E_{c \in D_c, x_0 \in p_\theta(x_0 | c)} [r(c, x_0)] - \beta KL(p_\theta(x_0 | c) || p_{ref} (x_0 | c))

\\=

max E [r(c, x_0)] - \beta \int_{x_0} p_\theta(x_0|c) log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)}) 

\\=

max [E [r(c, x_0)] - \beta E[ log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)}) ]]

\\=

max [E [r(c, x_0) - \beta log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)})]

\\=

min [E [log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)}) - \frac{1}{\beta} r(c, x_0)]

\\=

min E [log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)}) - log(exp(\frac{1}{\beta} r(c, x_0)))]

\\=

min E [log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))})]
$$
可以看出，RLHF的目标函数最终可以解释为：

- 最终推导出来的式子与KL散度的表达式相似。
- 让待优化模型 $p_\theta (x_0 | c)$ 与分布 $p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))$ 的KL散度等于0 （形式上是，但后者的分布并不满足概率分布的条件，还需要加入配分项）
- 由于 $\int_{x_0} p_{ref} (x_0 | c) = 1$ ，所以显然   $\int_{x_0} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0)) \ne 1$ 。然而待优化模型是一个概率密度函数，必须要保证 $\int p_\theta (x_0 | c) = 1$ 。因此，需要给 $p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))$ 加入配分项，使得 $\int_{x_0} \frac{1}{Z(c)} p_{ref} (x_0 | c) exp(\frac{1}{\beta} r(c, x_0)) = 1$ 。
- 显然， $Z(c) = \int_{x_0} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))$ 

因此，上式可以等价替换成：
$$
max_{p_\theta} E_{c \in D_c, x_0 \in p_\theta(x_0 | c)} [r(c, x_0)] - \beta KL(p_\theta(x_0 | c) || p_{ref} (x_0 | c))

\\=

min E [log(\frac{p_\theta(x_0|c)}{p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))})]

\\=

min E [log(\frac{p_\theta(x_0|c)}{\frac{1}{Z(c)} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))}) - log(Z(c))]
$$
待优化模型的目标就是 $\frac{1}{Z(c)} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))$ ，令 $p_\theta(x_0 | c)^* = \frac{1}{Z(c)} p_{ref} (x_0 | c)log(exp(\frac{1}{\beta} r(c, x_0)))$

上式可以表示为：
$$
min E [log(\frac{p_\theta(x_0|c)}{\frac{1}{Z(c)} p_{ref} (x_0 | c)log(exp(\frac{1}{\beta} r(c, x_0)))}) - log(Z(c))]
$$
由于 $Z(c)$ 与最优化目标 $p_\theta$ 无关，因此上式等价于最小化第一项，即：
$$
min E [log(\frac{p_\theta(x_0|c)}{\frac{1}{Z(c)} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))})]

\\=

min E [log(\frac{p_\theta(x_0|c)}{p_\theta(x_0|c)^*})]

\\=

min KL(p_\theta(x_0|c) || p_\theta(x_0|c)^*)
$$
其中
$$
p_\theta(x_0 | c)^* = \frac{1}{Z(c)} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))
$$
# 4 DPO for Diffusion Models

定义：

- 一个固定的数据集 $D = \{ (c, x_0^w, x_0^l) \}$ ，数据是从模型 $p_{ref}$ 中用 prompt $c$ 生成的，并且经过人类标注 $x_0^w > x_0^l$ 
- 我们的目标是学习一个新的模型 $p_\theta$ 来对齐人类先验。
- 为了实现该目标，使用DPO算法，并且推导出了不需要奖励函数的目标函数 $p_\theta(x_0 | c)^* = \frac{1}{Z(c)} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))$ 
- 然而，目标函数看似简单，但在扩散模型中却需要从 $x_T$ 逐渐去噪到 $x_0$ 才能够计算，耗时巨大。

## 4.1 蒸馏 $x_0$ 等价于蒸馏任意 $x_t$

$$
KL(p_\theta(x_{0:T} | c) || p_{ref} (x_{0:T} | c))

\\=

E log(\frac{p_\theta(x_{0:T} | c)}{p_{ref} (x_{0:T} | c)})

\\=

E log(\frac{ p_\theta(x_T|c) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t, c)}{p_{ref}(x_T|c) \prod_{t=1}^{T} p_{ref}(x_{t-1} | x_t, c)})

\\=

E log(\frac{ p_\theta(x_0|x_1, c) p_\theta(x_T|c) \prod_{t=2}^{T} p_\theta(x_{t-1} | x_t, c)}{p_{ref}(x_0|x_1, c) p_{ref}(x_T|c) \prod_{t=2}^{T} p_{ref}(x_{t-1} | x_t, c)})
$$

由于：
$$
p_\theta (x_0 | x_1, c) = \frac{p_\theta(c) p_\theta(x_0|c) p_\theta(x_1 | c, x_0)}{p_\theta(c)p(x_1|c)}
=
\frac{p_\theta(x_0|c) p_\theta(x_1 | c, x_0)}{p_\theta(x_1|c)}

\\
p_{ref} (x_0 | x_1, c) = \frac{p_{ref}(c) p_{ref}(x_0|c) p_{ref}(x_1 | c, x_0)}{p_{ref}(c)p_{ref}(x_1|c)}
=
\frac{p_{ref}(x_0|c) p_{ref}(x_1 | c, x_0)}{p_{ref}(x_1|c)}
$$
代入：
$$
KL(p_\theta(x_{0:T} | c) || p_{ref} (x_{0:T} | c))

\\=

E log(
\frac{ \frac{p_\theta(x_0|c) p_\theta(x_1 | c, x_0)}{p_\theta(x_1|c)} p_\theta(x_T|c) \prod_{t=2}^{T} p_\theta(x_{t-1} | x_t, c)}
{\frac{p_{ref}(x_0|c) p_{ref}(x_1 | c, x_0)}{p_{ref}(x_1|c)} p_{ref}(x_T|c) \prod_{t=2}^{T} p_{ref}(x_{t-1} | x_t, c)})

\\=

E log(
\frac
{p_{ref}(x_1|c) p_\theta(x_0|c) p_\theta(x_1 | c, x_0) p_\theta(x_T|c) \prod_{t=2}^{T} p_\theta(x_{t-1} | x_t, c)}
{p_\theta(x_1|c) p_{ref}(x_0|c) p_{ref}(x_1 | c, x_0) p_{ref}(x_T|c) \prod_{t=2}^{T} p_{ref}(x_{t-1} | x_t, c)})

\\=
E 
[
log(\frac{p_\theta(x_0|c)}{p_{ref}(x_0|c)})
+
log(\frac{p_{ref}(x_1|c)}{p_\theta(x_1|c)})
+
log(\frac{p_\theta(x_1 | c, x_0)}{p_{ref}(x_1 | c, x_0)})
+
\sum_{t=2}^{T} log(\frac{ p_\theta(x_{t-1} | x_t, c)}{p_{ref}(x_{t-1} | x_t, c)})
]

\\=
E 
[
log(\frac{p_\theta(x_0|c)}{p_{ref}(x_0|c)})
]
+
KL(...) + KL(...) + KL(...)


\\

\ge

E 
[
log(\frac{p_\theta(x_0|c)}{p_{ref}(x_0|c)})
]

=

KL(p_\theta(x_{0} | c) || p_{ref} (x_{0} | c))
$$
因此，
$$
KL(p_\theta(x_{0:T} | c) || p_{ref} (x_{0:T} | c))

\ge

KL(p_\theta(x_{0} | c) || p_{ref} (x_{0} | c))
$$
即，最大化 $KL(p_\theta(x_{0} | c) || p_{ref} (x_{0} | c))$ 等价于最大化 $KL(p_\theta(x_{0:T} | c) || p_{ref} (x_{0:T} | c))$ 。

---

## 4.2 对 $x_0$ 计算奖励函数等价于对任意 $x_t$ 计算奖励函数

由刚才推导的:
$$
p_\theta(x_0 | c)^* = \frac{1}{Z(c)} p_{ref} (x_0 | c)exp(\frac{1}{\beta} r(c, x_0))
$$
可知，
$$
r(c, x_0) = \beta log(\frac{Z(c) p_\theta(x_0 | c)^*}{p_{ref} (x_0 | c)})

\\=

\beta log(\frac{p_\theta(x_0 | c)^*}{p_{ref} (x_0 | c)}) + \beta log(Z(c))
$$
在扩散模型中，$x_0$ 的计算是一个马尔科夫链：
$$
r(c, x_0) = \beta log(\frac{p_\theta(x_0 | c)^*}{p_{ref} (x_0 | c)}) + \beta log(Z(c))

\\=

\beta log
[

\frac
{\prod_{t=1}^{T} p_\theta(x_{t-1} | x_0, x_t, c)^*}
{\prod_{t=1}^{T} p_{ref}(x_{t-1} | x_0, x_t, c)}

]

+ 
\beta log(Z(c))

\\=

\beta log
[
\prod_{t=1}^{T} 
\frac
{p_\theta(x_{t-1} | x_0, x_t, c)^*}
{p_{ref}(x_{t-1} | x_0, x_t, c)}

]

+ 
\beta log(Z(c))

\\=

\beta 
\sum_{t=1}^{T} 
log
[
\frac
{p_\theta(x_{t-1} | x_0, x_t, c)^*}
{p_{ref}(x_{t-1} | x_0, x_t, c)}

]

+ 
\beta log(Z(c))
$$
对于奖励函数的损失函数，由琴生不等式
$$
f(\frac{x_1 + x_2 + ... + x_n}{n}) \le \frac{f(x_1) + f(x_2) + ... + f(x_n)}{n}
\\
f(E(x)) \le E(f(x))
$$
可知 （由于有负号，所以这里大小关系是反的）：
$$
L_{BT} (\phi) = - E_{c, x_{0}^w , x_{0}^l } [log(\sigma (r(c, x_0^w) - r(c, x_0^l)))]
\\ \le \\
-  log
[
\sigma [E_{c, x_{0}^w , x_{0}^l } [r(c, x_0^w) - r(c, x_0^l)]]
]
$$
把 $r(c, x_0)$ 带入到奖励函数的损失函数，可知：
$$
L_{BT} (\phi) \le

-  log
[
\sigma [E_{c, x_{0}^w , x_{0}^l } [r(c, x_0^w) - r(c, x_0^l)]]
]

\\=

-  log
[
\sigma [

E_{c, x_{0}^w , x_{0}^l } [

\beta 
\sum_{t=1}^{T} 

log
[
\frac
{p_\theta(x_{t-1} | x_0^w, x_t^w, c)^*}
{p_{ref}(x_{t-1} | x_0^w, x_t^w, c)}
]
-
log
[
\frac
{p_\theta(x_{t-1} | x_0^l, x_t^l, c)^*}
{p_{ref}(x_{t-1} | x_0^l, x_t^l, c)}
]

]

]
]

\\=

-  log
[
\sigma [

E_{c, x_{0}^w , x_{0}^l } [

\beta 
T

E_{t}

[
log
[
\frac
{p_\theta(x_{t-1} | x_0^w, x_t^w, c)^*}
{p_{ref}(x_{t-1} | x_0^w, x_t^w, c)}
]
-
log
[
\frac
{p_\theta(x_{t-1} | x_0^l, x_t^l, c)^*}
{p_{ref}(x_{t-1} | x_0^l, x_t^l, c)}
]
]

]

]
]

\\=

-  log
[
\sigma [

\beta 
T
E_{c, x_{t}^w , x_{t}^l } 



E_{t, x_t^w, x_t^l}

[
log
[
\frac
{p_\theta(x_{t-1} | x_0^w, x_t^w, c)^*}
{p_{ref}(x_{t-1} | x_0^w, x_t^w, c)}
]
-
log
[
\frac
{p_\theta(x_{t-1} | x_0^l, x_t^l, c)^*}
{p_{ref}(x_{t-1} | x_0^l, x_t^l, c)}
]
]



]
]

\\ = \\

-  log
[
\sigma [

\beta 
T
E_{c, x_{t}^w , x_{t}^l } 



E_{t, x_t^w, x_t^l}

[
log
[
\frac
{p_\theta(x_{t-1} | x_0^w, x_t^w, c)^*}
{q(x_{t-1} | x_0^w, x_t^w, c)}

\frac
{q(x_{t-1} | x_0^w, x_t^w, c)^*}
{p_{ref}(x_{t-1} | x_0^w, x_t^w, c)}
]
-
log
[
\frac
{p_{\theta}(x_{t-1} | x_0^l, x_t^l, c)^*}
{q(x_{t-1} | x_0^l, x_t^l, c)}

\frac
{q(x_{t-1} | x_0^l, x_t^l, c)^*}
{p_{ref}(x_{t-1} | x_0^l, x_t^l, c)}
]
]
]
]

\\ = \\

-  log
[
\sigma [

\beta 
T

E_{t, x_t^w, x_t^l}

[

\\
KL(p_\theta(x_{t-1} | x_0^w, x_t^w, c)^* || q(x_{t-1} | x_0^w, x_t^w, c)) \\
+
KL(q(x_{t-1} | x_0^w, x_t^w, c)^* || p_{ref}(x_{t-1} | x_0^w, x_t^w, c)) \\

-
KL(p_\theta(x_{t-1} | x_0^l, x_t^l, c)^* || q(x_{t-1} | x_0^l, x_t^l, c)) \\

-

KL(q(x_{t-1} | x_0^l, x_t^l, c)^* || p_{ref}(x_{t-1} | x_0^l, x_t^l, c)) \\

]
]
$$
由：
$$
E(log(\frac{p}{q})) = E(-log((\frac{q}{p})^{-1})) = KL(p || q) = -KL(q || p)
$$
可知，把 $q(...)$ 都提到前面来：
$$
L_{BT} (\phi) \le \\


-  log
[
\sigma [

\beta 
T

E_{t, x_t^w, x_t^l}

[

\\
- KL(q(x_{t-1^w} | x_0^w, x_t^w, c) || p_\theta(x_{t-1}^w | x_0^w, x_t^w, c)^*) \\
+
KL(q(x_{t-1}^w | x_0^w, x_t^w, c)^* || p_{ref}(x_{t-1}^w | x_0^w, x_t^w, c)) \\

+
KL(q(x_{t-1}^l | x_0^l, x_t^l, c)^* || p_{\theta}(x_{t-1}^l | x_0^l, x_t^l, c)) \\

-

KL(q(x_{t-1}^l | x_0^l, x_t^l, c)^* || q_{ref}(x_{t-1}^l | x_0^l, x_t^l, c))

]
]
$$
根据DDPM的推导，KL散度等价于预测噪声的 $l2$ 损失：
$$
L_{BT} (\phi) \le \\


-  log
[
\sigma [

\beta 
T

E_{t, x_t^w, x_t^l}

[

\\
- || \epsilon ^w - \epsilon_{\theta}(x_t^w, t)^*)||^2 \\
+
||\epsilon^w - \epsilon_{ref}(x_t^w, t)||^2 \\

+
||\epsilon^l - \epsilon_{\theta}(x_t^l, t)||^2 \\

-

||\epsilon^l - \epsilon_{ref}(x_t^l, t)||^2

]
]
$$
其中，

- $\epsilon ^w , \epsilon ^l \in N(0, I)$
- 由于 $\beta$ 是RLHF中控制蒸馏损失的权重，因此可以把 $T$ 合并到 $\beta$ 中

合并一下：
$$
L_{BT} (\phi) \le \\


-  log
[
\sigma [

\beta 
T

E_{t, x_t^w, x_t^l}

[


(||\epsilon^w - \epsilon_{ref}(x_t^w, t)||^2 - || \epsilon^w - \epsilon_{\theta}(x_t^w, t)^*)||^2)
-
(||\epsilon^l - \epsilon_{ref}(x_t^l, t)||^2 - ||\epsilon^l - \epsilon_{\theta}(x_t^l, t)||^2)

]
]

\\ = \\
-  log
[
\sigma [

- \beta 

E_{t, x_t^w, x_t^l}

[


(|| \epsilon^w - \epsilon_{\theta}(x_t^w, t)^*)||^2 - ||\epsilon^w - \epsilon_{ref}(x_t^w, t)||^2)
-
(||\epsilon^l - \epsilon_{\theta}(x_t^l, t)||^2 - ||\epsilon^l - \epsilon_{ref}(x_t^l, t)||^2)

]
]
$$
如何理解：

- $\epsilon^w, \epsilon^l$ 都是同一个采样的噪声，完全相同。

- 需要最小化 $L_{BT}$ ，就需要最大化 $log[\sigma(...)]$ ，就需要最大化 $-\beta E[...]$ ，就需要最小化 $ () - ()$ ，就要保证：
  $$
  (|| \epsilon^w - \epsilon_{\theta}(x_t^w, t)^*)||^2 - ||\epsilon^w - \epsilon_{ref}(x_t^w, t)||^2)
  \lt
  (||\epsilon^l - \epsilon_{\theta}(x_t^l, t)||^2 - ||\epsilon^l - \epsilon_{ref}(x_t^l, t)||^2)
  $$

- 等价于
  $$
  (|| \epsilon^w - \epsilon_{\theta}(x_t^w, t)^*)||^2 - ||\epsilon^l - \epsilon_{\theta}(x_t^l, t)||^2)
  \lt
  (||\epsilon^w - \epsilon_{ref}(x_t^w, t)||^2 - ||\epsilon^l - \epsilon_{ref}(x_t^l, t)||^2)
  $$

- 对于两张图像，经过多次优化之后，右侧 $\epsilon_{ref}$ 的值在多次优化中都不变，而左边需要越来越小。意味着 $\epsilon(x_t^w, t)$ 需要与 gt噪声 $\epsilon$ 更接近（更接近0，因为l2损失>=0）。

# 5 实验

## 5.1 实验设置

- 模型
  - SD1.5，SDXL-1.0
- 数据集
  - Pick-a-Pic数据集，图像来自于SDXL-beta，Dreamlike和微调SD1.5生成的数据。使用Pick-a-Pic v2数据集。排除 ∼12% 的数据对后，我们最终得到 851,293 对，其中有 58,960 个独一的prompt。
- 超参
  - 优化器：AdamW for SD1.5 ，Adafactor for SDXL来节省内存
  - 2048 （数据对）作为batchsize （实际是4096？）
  - 16个A100GPU，local bs=1, 梯度累计是128
  - 使用固定方形的分辨率训练（下载 pick-a-pick数据之后发现，分辨率既有512也有1024，所以也可以理解为什么bs=1）
  - 学习率 $\frac{2000}{\beta} 2.048 \times 10^{-8}$ ，前 $25\%$ 的阶段使用线性warmup
  - 对于SD1.5和SDXL，都发现当 $\beta \in [2000, 5000]$ 时的效果最好。最终SD1.5使用 $\beta=2000$ , SDXL的 $\beta = 5000$ 
  - 训练1000 - 2000 steps

# 6 伪代码

```python
def loss(model, ref_model, x_w, x_l, c, beta):
    """
    # This is an example psuedo-code snippet for calculating the Diffusion-DPO loss
    # on a single image pair with corresponding caption
    model: Diffusion model that accepts prompt conditioning c and time conditioning t
    ref_model: Frozen initialization of model
    x_w: Preferred Image (latents in this work)
    x_l: Non-Preferred Image (latents in this work)
    c: Conditioning (text in this work)
    beta: Regularization Parameter
    returns: DPO loss value
    """
    timestep = torch.randint(0, 1000)
    noise = torch.randn_like(x_w)
    noisy_x_w = add_noise(x_w, noise, t)
    noisy_x_l = add_noise(x_l, noise, t)
    model_w_pred = model(noisy_x_w, c, t)
    model_l_pred = model(noisy_x_l, c, t)
    ref_w_pred = ref(noisy_x_w, c, t)
    ref_l_pred = ref(noisy_x_l, c, t)

    model_w_err = (model_w_pred - noise).norm().pow(2)
    model_l_err = (model_l_pred - noise).norm().pow(2)
    ref_w_err = (ref_w_pred - noise).norm().pow(2)
    ref_l_err = (ref_l_pred - noise).norm().pow(2)
    w_diff = model_w_err - ref_w_err
    l_diff = model_l_err - ref_l_err
    inside_term = -1 * beta * (w_diff - l_diff)
    loss = -1 * log(sigmoid(inside_term))
    return loss
```

