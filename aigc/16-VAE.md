Auto-Encoding Variational Bayes



# 1 介绍

解决的问题：如何在大型数据集提取的连续隐变量中，高效的学习后验概率，从而建模有向概率模型。

贡献主要包括两点：

- 重参数化的变分下界能够产生一个更低的边界估计，可以直接使用标准的随机梯度下降方法进行优化。
- 对于每个点的隐变量都是连续的独立同分布的数据集，通过使用所提出的下界估计，将近似推理模型(也称为识别模型)拟合后验概率分布，可以使后验推理特别有效

# 2 方法

## 2.1 问题描述

数据集 $X = \{x^{i} \}_{i=1}^N$  包含 $N$ 个独立同分布的样本。我们假设数据集是从一些包含不可观测的连续型随机变量 $z$ 中随机生成的，该过程包含两个步骤：

- (1) 从先验分布 $p(z)$ 中采样得到 $z^{(i)}$ 
- (2) 从$p(x|z)$ 中生成 $x$ 

## 2.2 变分边界



**1. 黑盒神经网络无所不能**

---

为了计算后验概率：
$$
p(z|x)= \frac{p(x, z)}{p(x)}
\\=
\frac{p(x|z)p(z)}{p(x)}
$$
其中：

- 隐变量 $p(z)$ 的先验分布是想要计算的目标---未知。
-  $p(x)$ 的先验分布就更加未知了（之所以计算后验概率 $p(z|x)$ ，就是因为 $p(x)$ 的概率未知且太复杂，而 $p(z|x)$ 通常可以假设为高斯分布）
- $p(x|z)$  由于即不知道 $x$ 的分布，也不知道 $z$ 的分布，所以不管直接算，还是用贝叶斯公式，都也是未知的。

想用贝叶斯公式，但右侧三项都是未知的。。。。

因此，才会使用变分推断来求解 $p(z|x)$ ：

- 找到一个分布 $q_\phi(z|x)$ ，来近似 $p(z|x)$ 

想法很简洁，但是如何找到 $q_\phi(z|x)$ ？这个问题过于棘手，所以就干脆交给神经网络来做，神经网络的参数是 $\phi$ 。





**2. 绕成了死结**

---

网络找到的分布 $q_\phi (z|x)$ 需要和后验分布 $p(z|x)$ 尽可能接近，很自然的想法是使用KL散度进行度量：
$$
D_{KL}(q_\phi(z|x) | p(z|x))
$$
根据 KL 散度的定义，可以展开为：
$$
D_{KL}(q_\phi(z|x) | p(z|x))
\\=
\int q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(z|x)}) dz
\\=
\int q_\phi (z|x) log(q_\phi (z|x)) dz
-
\int q_\phi (z|x) log(p(z|x)) dz
$$
然而，为了计算预测分布和真实分布 $p(z|x)$ 的分布，怎么也不可能绕开真实分布 $p(z|x)$ ，否则和谁去计算KL散度？

矛盾的地方是，我们想估计 $q_\phi(z|x)$ 来代替 $p(z|x)$ ，但是用 KL 散度让两者分布接近时，又需要 $p(z|x)$ ，绕成了死结。





**3. 想方设法替换掉 $p(z|x)$**

---

想要替换掉 $p(z|x)$ ，貌似只能使用下式：
$$
p(z|x) = \frac{p(x, z)}{p(x)}
$$
先代入进去看看：
$$
D_{KL}(q_\phi(z|x) | p(z|x))
\\=
\int q_\phi (z|x) log(q_\phi (z|x)) dz
-
\int q_\phi (z|x) log(p(z|x)) dz

\\=

\int q_\phi (z|x) log(q_\phi (z|x)) dz
-
\int q_\phi (z|x) log(\frac{p(x, z)}{p(x)})) dz

\\=

\int q_\phi (z|x) log(q_\phi (z|x)) dz
-
\int q_\phi (z|x)log(p(x, z)) dz
+ 
\int q_\phi (z|x)log(p(x)) dz
$$
继续往下推好像没有意义了。。。但是可以看出，上式中的前两个积分都是形如；
$$
\int q(z) f(z, \cdot) dz
$$
可以看出，上式就是期望的定义。所以换成期望再试试：
$$
D_{KL}(q_\phi(z|x) | p(z|x))
\\=
\int q_\phi (z|x) log(q_\phi (z|x)) dz
-
\int q_\phi (z|x)log(p(x, z)) dz
+ 
\int q_\phi (z|x)log(p(x)) dz
\\=

E_{q_\phi(z|x)}[log(q_\phi(z|x)]
-
E_{q_\phi(z|x)}[log(p(x, z))]
+
\int q_\phi (z|x)log(p(x)) dz
$$
只剩下第三项是积分了，但是对 $z$ 求积分，不关 $log(p(x))$ 的事情，所以 $log(p(x))$ 可以提到积分外部：
$$
D_{KL}(q_\phi(z|x) | p(z|x))
\\=
E_{q_\phi(z|x)}[log(q_\phi(z|x)]
-
E_{q_\phi(z|x)}[log(p(x, z))]
+
log(p(x)) \int q_\phi (z|x) dz
$$
剩下的 $\int q_\phi (z|x) dz$ 在实数域内对一个概率分布 $q_\phi(z|x)$ 求积分，概率和为1，所以积分结果就是1。与是不是条件概率分布无关，因此：
$$
D_{KL}(q_\phi(z|x) | p(z|x))
\\=
E_{q_\phi(z|x)}[log(q_\phi(z|x)]
-
E_{q_\phi(z|x)}[log(p(x, z))]
+
log(p(x))
$$


**4. 峰回路转**

---

生成模型中，如果获得了数据分布 $p(x)$ 就是无敌的存在，直接掌握了任意真实图像生成的分布函数。然而由于 $p(x)$ 的分布过于复杂且未知，才考虑去找到一个简单的隐变量分布函数 $p(z|x)$ 。

- 使用 $q_\phi(z|x)$ 来估计 $p(z|x)$ 
- 使用 KL 散度来使两个后验概率分布接近
- 想方设法去掉 KL 散度中的真实后验概率分布 $p(z|x)$
- 最终上式 KL 散度的式子，虽然借助强大的黑盒神经网络帮我们预测出 $log(q_\phi(z|x)$ 替换掉了 $p(z|x)$ ，但又引入了 $p(x)$  ---- 如果能知道 $p(x)$ ，直接采样图像就可以了，干嘛还要绕这么一大圈？

所以，上式又无解了。。。

不过可以反过来想一想，表达式中出现了 $log(p(x))$ ：
$$
log(p(x)) = 

D_{KL}(q_\phi(z|x) | p(z|x))
-
E_{q_\phi(z|x)}[log(q_\phi(z|x)]
+
E_{q_\phi(z|x)}[log(p(x, z))]
$$

- 我们最初的出发点是想找个隐变量 $z$ 来解决真实分布 $p(x)$ 过于复杂的问题
- 绕了一大圈，反而把 $p(x)$ 的log形式 $log(p(x))$ 给找出来 了
- 有了表达式，直接极大似然估计就可以了！ 而且 $log(p(x))$ 就是对数似然。接下来对于观测样本，我们只需要保证上式右侧尽可能大就行了。

观察上式等式右侧，$D_{KL}$ 是 0-1之间，永远非负。因此:
$$
log(p(x)) \ge
-
E_{q_\phi(z|x)}[log(q_\phi(z|x)]
+
E_{q_\phi(z|x)}[log(p(x, z))]
$$
所以：
$$
ELBO =
-
E_{q_\phi(z|x)}[log(q_\phi(z|x)]
+
E_{q_\phi(z|x)}[log(p(x, z))]
$$
被称为证据 $log(p(x))$ 的下界。为了极大化 $log(p(x))$ ，只需极大化 $ELBO$ ：

- $ELBO$ 中还剩下 $p(x, z)$ 未知



**5. 最后一公里**

---

$ELBO$ 中涉及到的 $p(x, z)$ 还未知，再借助全概率公式 $p(x, z) = p(z|x)p(x)$
$$
ELBO =
-
E_{q_\phi(z|x)}[log(q_\phi(z|x)]
+
E_{q_\phi(z|x)}[log(p(x, z))]

\\=
E_{q_\phi(z|x)}[log(\frac{p(z|x)p(x)}{q_\phi(z|x)})]

\\=

E_{q_\phi(z|x)}[log(p(x) \frac{p(z|x)}{q_\phi(z|x)})]

\\=

E_{q_\phi(z|x)}[log(p(x)) + log(\frac{p(z|x)}{q_\phi(z|x)})]

\\=
E_{q_\phi(z|x)}[log(p(x))]
+
E_{q_\phi(z|x)}[log(\frac{p(z|x)}{q_\phi(z|x)})]

\\=
E_{q_\phi(z|x)}[log(p(x))]
+
E_{q_\phi(z|x)}[log(\frac{p(z|x)}{q_\phi(z|x)})]
$$
根据 KL 散度的定义：
$$
E_{q_\phi(z|x)}[log(\frac{p(z|x)}{q_\phi(z|x)})]
\\=
\int_{q_\phi(z|x)} q(z) log(\frac{p(z|x)}{q_\phi(z|x)}) dz

\\=

- D_{KL}(q_\phi(z|x), p(z|x))
$$




---

最终，需要用到的 ELBO 的形式为 ：
$$
ELBO = - D_{KL}(q_\phi(z|x), p(z|x)) + E_{q_\phi(z|x)}[log(p(x))]
$$

- $p(x)$ 暂时仍未知，论文中是condition on z （之后还是使用盲盒 decoder 来预测分布），即：
  $$
  ELBO = - D_{KL}(q_\phi(z|x), p(z|x)) + E_{q_\phi(z|x)}[log(p(x|z))]
  $$
  
- 这里对于为什么能够使用 $p(x)$ 代替 $p(x|z)$ 的解释如下：
  -  $p(z|x)$ 是我们想要找到的隐变量的分布，直接找比较困难，所以使用后验分布 $p(z|x)$ 来代替 $p(x)$ 。而如果模型优化的足够好，可以认为 $p(z|x) = p(z)$ 。所以根据贝叶斯定理，有 $p(x) = p(x|z)$ 。

 ## 2.3 蒙特卡罗期望估计

具体理解在对应的笔记里。这里只介绍本文需要用到的条件和结论：

- 对于非常复杂的概率密度函数 $f(x)$ 求积分 $\int_a^b f(x) dx$ 可能是非常复杂的

- 把该问题转化为 :
  $$
  \int_a^b f(x) dx = \int_a^b \frac{f(x)}{p(x)} p(x) dx
  \\=
  \frac{1}{n} \sum_i \frac{f(x_i)}{p(x_i)}
  \\=
  E_{p(x)} [\frac{f(x_i)}{p(x_i)}]
  $$
  其中，$p(x)$ 是一个更简单的密度函数，且具有对于任意 $p(x)$ ，总有 $p(x) >= f(x)$ 。$x_i$ 是从 $p(x)$ 中采样出的样本。

## 2.3 SGVB 和 AEVB 算法

- SGVB : 随机梯度变分贝叶斯估计，ELBO就是 SGVB 的版本之一
- AEVB ：变分贝叶斯自编码器

- 二者的关系：AEVB是一种算法，该算法的基础是SGVB

有了神经网络替代的 $q_\phi (z|x)$ 之后，就可以从中采样出 $z$ 了（利用重参数法）：
$$
\tilde{z} = g_\phi(\epsilon, x)
$$
其中， $\epsilon \in p(\epsilon)$ 是重参数法中的噪声分布。$g_\phi(\epsilon, x)$ 表示：从输入$x$ ，经过一系列函数（encoder，采样等），涉及到的模型参数为 $\phi$ ，采样噪声为 $\epsilon$ 。总是就是无论无何，给一个 $x$ ，一个参数为 $\phi$ 的网络用来估计后验概率分布，一个采样噪声 $\epsilon$ ，$g$ 函数就可以最终采样出一个隐编码 $\tilde{z}$ 。

ELBO 有两种表达式，这里使用 $ELBO = - D_{KL}(q_\phi(z|x), p(z|x)) + E_{q_\phi(z|x)}[log(p(x|z))]$ 。

则 $ELBO$ 的期望项可以用蒙特卡洛期望估计进行替换：
$$
ELBO = - D_{KL}(q_\phi(z|x), p(z|x)) + \frac{1}{L} \sum_{l=1}^{L} log(p(x_i|z_{i,l}))
$$
其中：

- 由于 $z$ 是 $q_\phi(z|x)$ 中采样得来的（采样的函数为 $\tilde{z} = g_\phi(\epsilon, x)$ ）。也就是说，对于同一个 $x_i$ ，可以采样若干个 $z_{i}$ ，每个 $z$ 的下标为 $l$ ，上式的求和是表示每个样本 $x_i$ 共采样出 $L$ 个 $z$ 出来。
- 上式 $ELBO$ 可以理解为：
  - 第一项是编码器的损失，希望编码器得到的隐变量的后验分布于真实后验分布一致
  - 第二项是解码器的损失，希望似然概率分布最大。

假设数据集 $X$ 中共有$N$ 个样本。每个batch从中随机采样 $M$ 个样本 $X^M = \{x_i\}_{i=1}^{M}$ ，每个样本又采样出 $L$ 个隐变量。

- 虽然此时 $p(z|x)$ 还未知，也不知道 $p(x_i|x_{i,l})$ 的分布和参数是哪些，也就没法进行极大似然估计。后面会进行各种假设。。。。。

- 但是如果 $ELBO$ 此时的各项都当作已知项时，我们就可以把 $ELBO$ 当作损失函数进行优化了：
  - KL散度越小越好，最小等于0.
  - 似然概率分布越大越好
- 上式 $ELBO$ 的优化方向是刚好相反的。而我们只需要计算 $ELBO$ 的梯度 $\nabla L = \nabla ELBO$ ，让参数沿着梯度相同的方向优化即可，这也是论文中的写法，没有给 ELBO 取反。

算法流程如下：

![image-20230627233537787](imgs/16-VAE/image-20230627233537787.png)

- 随机初始化编码器，解码器的参数 $\theta, \phi$ 
- 从数据集中采样出 $M$ 个样本
- 为了重采样，从 $p(\epsilon)$ 中采样出一个噪声 $\epsilon$ 。（噪声的分布未知，后面也会圆回来）
- 根据 ELBO 计算梯度 $\nabla L$ ，得到梯度 $g$ 
- 使用 SGD 等梯度下降法更新  $\theta, \phi$ 
- 重复，直至收敛

所有数据训练一遍，总的 ELBO 为：
$$
ELBO = N \frac{1}{M}\sum_{i=1}^M \frac{1}{L} \sum_{l=1}^{L}  log(p(x_i|z_{i,l}))
$$

# 3 Variational Auto-Encoder

具体细节有时间再推，其实也没什么细节。就是对之前埋的一些坑进行各种假设：
$$
ELBO = - D_{KL}(q_\phi(z|x), p(z|x)) + \frac{1}{L} \sum_{l=1}^{L} log(p(x_i|z_{i,l}))
$$

- KL散度中，假设 $p(z|x)$ 是标准正态分布，$p(z) = N(z; 0, I)$
- 网络预测的后验概率分布 $q_\phi(z|x)$ 是高斯分布：$log(q_\phi(z|x_i)) = logN(z; \mu_i, \sigma^2_iI)$
- KL散度其实就是让模型预测的后验概率分布的均值是0，方差是1，从而接近真实 $p(z|x)$ 分布。后续推导这个。
- 真实似然分布 $p(x|z)$ 假设是高斯分布（或伯努利分布），从而进行极大似然估计。后续也会推导这个

## 3.1 两个高斯分布算KL散度

https://zhuanlan.zhihu.com/p/55778595

![image-20230628001608509](imgs/16-VAE/image-20230628001608509.png)

以两个高斯分布 $q(z), p(z)$ 为符号进行推导，$q(z) \sim N(\mu_1, \sigma_1^2), p(z) \sim N(\mu_2, \sigma_2^2)$ ，变量 $z$ 的维度是 $J$ 。
$$
D_{KL} (q(z)||p(z)) = \int q(z) log(\frac{q(z)}{p(z)}) dz

\\=
\int q(z) [log(q(z)) - log(p(z))] dz

\\=
\int q(z) log(q(z)) dz - \int q(z) log(p(z)) dz
$$
下面分别介绍左右两项。

### 3.2.1 计算第一项

$$
\int q(z) log(q(z)) dz
\\=
E_{q(z)} log(q(z))
\\=
E_{q(z)} [log  N(z; \mu_1, \sigma_1^2)]
\\=
$$



### 3.2.2 计算第二项



## 3.2 Gaussian MLP as Encoder or Decoder

![image-20230628001721798](imgs/16-VAE/image-20230628001721798.png)

## 3.3 Loss 计算示例



