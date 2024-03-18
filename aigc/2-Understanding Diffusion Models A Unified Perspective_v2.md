# Understanding Diffusion Models: A Unified Perspective

# 1 引用

- 论文：https://arxiv.org/abs/2208.11970

# 2 介绍：生成模型

对于观测到的感兴趣的样本 $x$，生成模型的目标是学习建模一个真实数据分布 $p(x)$。一旦学习到之后，就可以从估计的模型中生成新的样本。

当前有几种知名的研究方向，简要介绍如下：

- **Generative Adversarial Networks (GANs) ：**建模复杂分布的采样过程，并通过对抗的方式进行学习。
- **Likelihood-based：**寻求学习到一个模型，使得改模型对观测到的样本据哟较高的似然。该类方法包括自回归模型，normalizing flows，以及变分自编码器VAEs。
- **基于能量模型（Energy-based modeling）：**学习任意灵活的能量函数，并进行归一化。这种方法与score-based的方法高度相关，后者不是学习对能量函数本身进行建模，而是将基于能量的模型的分数学习为神经网络。

本文介绍扩散模型，并证明扩散模型即具有似然角度的解释，也具有score-based的角度的解释。

# 3 背景：ELBO, VAE, Hierarchical VAE

可以认为观测到的样本都是从一个看不见的隐变量中生成的，隐变量定义成 $z$。如，一群被锁起来只能看到墙上的二维影子的人，他们所看到的影子是其看不到的三维空间中由火焰映射出来的，对于这群人来说，他们观测到的影子是从更高纬度的抽象概念中生成的。

> 换一种熟悉的方式理解。可以认为观测到的样本 $x$ 都是来自于某个分布 $p_{\theta}(x)$ 的，其中 $\theta$ 是分布函数的参数。为了找到这个分布：

> 1. 频率派认为 $\theta$ 是一个固定值，为了从已经观测到的样本中估计 $\theta$，通常用极大似然估计，似然函数为 $p(x|\theta)$，求解目标为 $\theta = argmax log(p(x| \theta)$，通常用最优化进行求解。

> 2. 贝叶斯派认为 $\theta$ 不是一个固定值，而是一个随机变量，也服从某种分布 ，为了从已经观测到的样本中求解后验概率分布 $p(\theta|x)$，通常用最大后验法 $\theta = argmax log(p(\theta|x))$。其中，后验概率分布 $p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}$ 中包含：(1) 似然概率 $p(x|\theta)$ 通常用极大似然估计求解，深度学习中，通常把该项求极大值作为目标函数，利用最优化如梯度下降法求解，如VAE的Decoder的重建损失，扩散模型的马尔科夫去噪过程进行采样等。(2) $p(\theta)$ 先验概率，如VAE中假设是标准高斯分布，扩散模型的 $x_T$ 也是标准高斯噪声。(3) $p(x)$ 最难求，在最大后验法中没有这一项，如果需要求，可以使用 $\int p(x, \theta) dx = \int p(x|\theta) p(\theta) dx$ 并用蒙特卡罗采样等方法求解。

这里的隐变量 $z$ 和通常所说的分布函数 $\theta$ 意义相同。

## 3.1 Evidence Lower Bound

如上所述，用贝叶斯法计算后验概率时，似然概率通常用极大似然估计求解，并通常体现在损失函数中；先验概率通常假设为高斯分布等简单分布函数；唯一难以求解的就是 $p(x)$，从定义上，$p(x)$ 有两种表示方式：

- 基于观测样本和隐变量的联合概率计算边缘概率分布：
  $$
  p(x) = \int_z p(x, z) dz
  $$
  
- 基于概率公式的链式法则
  $$
  p(x) = \frac{p(x,z)}{p(z|x)}
  $$

所以，为什么说 $p(x)$ 难以求解：

- 如果通过边缘概率分布的方式求解，如果 $z$ 的维度很大，则需要 $dz_1 dz_2 dz_3, ....dz_n$，难以求解。
- 如果通过链式法则求解，又需要提前知道后验概率分布 $p(z|x)$，而后验概率分布是我们最终想要求解的目标，死循环。

所以就需要ELBO来近似求解。推导ELBO之前，先反过来看以下VAE和HMVAE是怎么计算 $p(x)$ 的。为了计算 $p(x)$ 中的 $p(x, z)$ 和 $p(z|x)$，都未知，但不同方法有不同的求解方式，可以解决这个问题，如：

- $p(x, z)$

  - VAE中，把 $p(x, z)$ 转化成 $p(z) p(x|z)$。对于先验概率分布 $p(z)$，直接假设为标准高斯分布。对于 $p(x|z)$，用神经网络 $q_{\theta} (x|z)$去拟合，并计算重建损失，即VAE的decoder。

  - HVAE中，用把 $p(x, z)$的采样过程替换成一个马尔科夫过程，实际转化方式也是转化成 $p(z) p(x|z)$。其中， $q_{\theta}$表示从隐变量 $z_t$到 $z_{t-1}$的分布的参数， $p(z_T)$通常假设是标准正态分布 ：
    $$
    p(x, z) = p(z_T) q_{\theta}(x | z_1) \prod_{t=2}^{T} q_{\theta}(z_{t-1} | z_t)
    $$

- $p(z|x)$

  - VAE中，用神经网络 $q_{\phi}(z|x)$ 去拟合，并与先验概率 $p(z)$ 计算先验匹配损失。

  - HVAE中， $p(z|x)$采样过程仍然是一个马尔科夫过程，其中， $\phi$是从隐变量 $z_{t-1}$到隐变量 $z_{t}$的分布的参数 ：
    $$
    p(z|x) = q_{\phi}(z_1 | x) \prod_{t=2}^{T} q_{\phi}(z_{t} | z_{t-1})
    $$

上面两个例子在推导ELBO之前提供了一个比较直观的理解，但是：

- 除了 VAE， HVAE，扩散模型等之外，有没有通用的后验概率求解方式？在没有对 $p(x, z), p(z|x)$ 做强假设的前提下，后验概率是无法直接求解的。
- 算 loss 时，有些地方用期望，有些地方用 KL散度，怎么来的？

所以，就需要推导出ELBO，ELBO不是用来估计 $p(x)$ 的，而是直接近似估计后验概率 $p(z|x)$ 的。 

从头梳理，贝叶斯方法中，最终目的是计算隐变量的分布即后验概率分布 $p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}$，然而， $p(x)$无法计算导致整个后验概率都无法求解。ELBO需要引入一个对后验分布的近似估计 $q_{\phi} (z|x)$来直接近似的估计真实后验概率。论文中的方式不便于理解，这里以另外一种方式进行推导：

估计的近似后验 $q_{\phi} (z|x)$ 需要与真实后验 $p(z|x)$ 尽可能接近，用KL散度衡量二者的相似程度：
$$
KL(q_{\phi}(z|x) || p(z|x)  = \int_z q_{\phi}(z|x) log (\frac{q_{\phi}(z|x)}{p(z|x)}) dz  = \sum_z q_{\phi}(z|x) log (\frac{q_{\phi}(z|x)}{p(z|x)})
$$
为了最小化 KL 散度，其中的真实后验 $p(z|x)$ 仍然是未知的，需要继续替换：
$$
p(z|x) = \frac{p(x, z)}{p(x)}
$$
即：
$$
KL(q_\phi (z|x)|| p(z|x))

\\= \int_z  q_\phi (z|x) log(\frac{p(x)q_{\phi}(z|x)}{p(x,z)}) dz

\\= \int_z  [q_\phi (z|x) [log(p(x)) + log(\frac{q_\phi (z|x)}{p(x,z)})] dz 

\\=  \int_z  q_\phi (z|x)log(p(x))dz + \int_z  q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(x,z)})dz

\\= log(p(x)) \int_z [q_{\phi}(z|x)dz + E[log(\frac{q_\phi (z|x)}{p(x, z)})]
$$
由于 $q_{\phi}(z|x)$ 是概率密度函数，积分是1，因此上式最终可以写为：
$$
KL(q_\phi (z|x)|| p(z|x)) = log(p(x)) + E[log(\frac{q_\phi (z|x)}{p(x, z)})]
$$
即，
$$
log(p(x)) = KL(q_\phi (z|x)|| p(z|x)) - E[log(\frac{q_\phi (z|x)}{p(x, z)})]
\\= KL(q_\phi (z|x)|| p(z|x)) + E[log(\frac{p(x, z) }{q_\phi (z|x)})]
$$
由于 KL散度恒大于等于0，因此：
$$
log(p(x)) \ge E[log(\frac{p(x, z) }{q_\phi (z|x)})] = ELBO
$$
即，ELBO是对数观测似然（证据）的下界。此时的目标就变成了极大化ELBO。

为什么极大化ELBO等价于极大化后验概率：

- ELBO中，本来想优化变分后验概率 $q_\phi (z|x)$ 来匹配真实后验概率 $p(z|x)$ ，该过程通过最小化KL散度（最小值是0）来实现。
- 不幸地是，KL散度需要真实后验概率 $p(z|x)$ ，然而我们并不知道，并且我们在贝叶斯方法中的最终目的就是求解真实后验概率。
- 然而，注意到上式中左侧的 $log(p(x))$ 是对数观测似然，是 $p(x, z)$ 的边缘概率分布，与变分后验概率 $q_\phi (z|x)$ 的参数 $\phi$ 无关，因此在观测样本给定时，$log(p(x))$ 不随模型参数 $\phi$ 而变化，在求解过程中，$log(p(x))$ 是不变的。
- 因此，为了最小化KL散度，等价于最大化 ELBO。

## 3.2 Variational Autoencoders

VAE的默认公式中，直接最大化ELBO，这种方法是变分的。但是ELBO中有一项 $p(x, z)$ 未知，如3.1节所介绍的示例，VAE中是把联合概率分布分解成：
$$
p(x, z) = p(z) p(x|z)
$$
因此ELBO变成：
$$
E[log(\frac{p(x, z) }{q_\phi (z|x)})] 
\\=
E[log(\frac{p(z) p(x|z)}{q_\phi (z|x)})]
\\=
E[log(p(x|z))] + E[log(\frac{p(z)}{q_\phi (z|x)})]
\\=
E[log(p(x|z))] + KL(p(z) || q_\phi (z|x))
\\=
E[log(p(x|z))] - KL(q_\phi (z|x) || p(z))
$$
从上式可以看出：

- 首先，学习到一个中间的bottleneck，输出分布是 $q_\phi (z|x)$ ，作为 encoder，把输入变换成一个隐空间的概率分布。
- 同时，学习一个确定性的函数 $p(x|z)$ 作为 decoder，把隐向量 $z$ 转变成 $x$ 。 

理解：

- 第一项是重建项，实际使用mse loss
- 第二项是先验分布匹配项，encoder预测均值和方差，并使其与标准正态分布对齐，实际使用的是多元高斯分布的kl损失，推导略。

## 3.3 Hierarchical Variational Autoencoders

HVAE是VAE的推广，具有多个层级的隐变量。每个隐变量都可以把所有之前的其他隐变量当作条件，但我们只关注一种特殊的 Markovian HVAE (MHVAE)。

在MHVAE中，生成过程是一个马尔科夫链，因此解码 $z_t$ 时，只依赖于前一个隐编码 $z_{t+1}$ 。直观上，这个过程可以看作时堆叠多个VAE。数学上可以MHVAE的后验概率表示成：
$$
q_\phi (z_{1:T}|x) = q_\phi(z_1 | x) \prod_{t=2}^{T} q_\phi(z_t | z_{t-1})
$$
MHVAE也是最大化ELBO $E[log(\frac{p(x, z) }{q_\phi (z|x)})] $ ，由于整个过程是马尔科夫链，因此：
$$
p(x, z_{1:T}) = p(z_T) p_\theta(x|z_1) \prod_{t=2}^T p_\theta (z_{t-1} | z_t)
$$
此时ELBO可以表示成：
$$
E[log(\frac{p(x, z) }{q_\phi (z|x)})] 
\\=
E[log(\frac{p(x, z_{1:T})}{q_\phi (z_{1:T}|x)})]

\\=

E[log( \frac{p(z_T) p_\theta(x|z_1) \prod_{t=2}^T p_\theta (z_{t-1} | z_t)} {q_\phi(z_1 | x) \prod_{t=2}^{T} q_\phi(z_t | z_{t-1})} )]
$$
推出上面的式子当作VAE到扩散模型的过渡。

#  4 Variational Diffusion Models

从HVAE到Variational Diffusion Model，需要添加三个关键约束条件：

- 隐变量的维度与数据维度完全相同
- 每个时间步的encoder （$z_{t-1} \to z_{t}$）不是通过学习得到的，而是预先定义的线性高斯模型。即，每一步生成隐变量的过程是确定的，且都是在上一步的基础上添加高斯噪声。需要注意，每一步 $x_t$ 的均值和方差是不固定的，均值是以上一步的输出为中心（$x_t \in N(\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I$）
- 最终 $T$ 时间步的隐编码是标准高斯分布。

---

- 从第一个约束条件，MHVAE对应的encoder可以改写成：
  $$
  q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
  $$

- 从第二个约束条件，每个时间步的编码器不是学习得到的，而是固定的线性高斯模型，其均值和方差可以是预先设置的超参，或预先学习好的超参。这里人为定义超参，第 $t$ 步的均值 $\mu_t (x_t) = \sqrt{\alpha_t} x_{t-1}$ ，方差 $\Sigma_t (x_t) = (1 - \alpha_t)I$ 。

  - 可以看出，均值是一定会随着 $x_{t-1}$ 的变化而变化的

  - 而如果不同时间步的 $\alpha_t$ 相同，则不同时间步的 $x_t$ 的方差也是相同的，因此，选择上述高斯分布的均值方差的形式的目的是确保不同时间步的隐变量的方差基本在一个大致相同的尺度，即 variance-preserving，保方差。

  - 为了提升灵活性，$\alpha_t$ 也可以随着时间步而变化。

  - 每个时间步的encoder $q(x_t | x_{t-1})$ 可以定义为：
    $$
    q(x_t | x_{t-1}) = N(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)I) 
    $$
    即，
    $$
    x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon
    $$
    其中，$\epsilon \in N(\epsilon; 0, I)$ 

- 从第三个约束条件，$\alpha_t$ 无论是固定的还是一种策略变化的，最终 $x_T$ 的分布都是标准高斯分布，即：
  $$
  p(x_T) = N(x_T; 0, I)
  $$

  - 这是由于 $x_t = \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1}} x_{0}  \sqrt{1 - \alpha_t\alpha_{t-1} \alpha_{t-2}...\alpha_{1}} \bar{z}_{0} =
    \sqrt{\bar{\alpha}_t} x_{0}  + \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}$ ，当 $\alpha_t < 1$ 且 $T$ 较大时，$x_0$ 的系数接近0，高斯噪声的系数接近1

上述过程如下图所示：

![image-20231204170201685](/home/mi/.config/Typora/typora-user-images/image-20231204170201685.png)

- 编码器的分布 $q(x_t | x_{t-1})$ 的参数不再是 $\phi$ ，而是在每个时间步定义的均值和方差。
- 因此，VDM中，只需要研究 $p_\theta (x_{t-1} | x_t)$ ，因此可以从高斯噪声生成新数据。

---

类似 MHVAE，ELBO可以表示成：
$$
E[log(\frac{p(x, z) }{q_\phi (z|x)})] 

\\=

E[log(\frac{p(x, z_{1:T}) }{q_\phi (z_{1:T}|x)})]
$$
由于 $x, z$ 此时维度相同，因此 $z_0 = x, 或 z_{1:T} = x_{1:T}$ ，且加噪过程没有可学习参数 $\phi$ 。即：
$$
E[log(\frac{p(x, z_{1:T}) }{q_\phi (z_{1:T}|x)})]

\\=

E[log(\frac{p(x_{0:T}) }{q (x_{1:T}|x_0)})]

\\=

E[log( \frac{p(x_T) \prod_{t=1}^T p_\theta (x_{t-1}| x_t) }{\prod_{t=1}^T q(x_t|x_{t-1})})]
$$
由于正向过程和逆向过程的最终目标分别是找到 $q(x_T|x_{T-1})$ （对照VAE的先验分布匹配项）和  $p_\theta (x_0|x_1)$  （对照VAE的重建项），因此先把这两项单独拆出来：
$$
ELBO =
E[log(\frac{p(x_T) p_\theta (x_0|x_1) \prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{q(x_T|x_{T-1}) \prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
\\=
E[log(\frac{p(x_T) p_\theta (x_0|x_1) }{q(x_T|x_{T-1}) )})] 
+
E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
\\=
E[log(p_\theta (x_0|x_1))] 
+
E[log(\frac{p(x_T) }{q(x_T|x_{T-1}) )})] 
+
E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
\\=
E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
+
E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]


\\=

E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
+
E[log(\frac{\prod_{t=1}^{T-1} p_\theta(x_{t}|x_{t+1})}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]

\\=

E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
+
E[\sum_{t=1}^{T-1} log(\frac{p_\theta (x_t | x_{t+1})}{q(x_t|x_{t-1})})]


\\=

E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
+
\sum_{t=1}^{T-1} E[ log(\frac{p_\theta (x_t | x_{t+1})}{q(x_t|x_{t-1})})]


\\=

E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
-
\sum_{t=1}^{T-1} E[KL( q(x_t|x_{t-1})  || p_\theta (x_t | x_{t+1}))]
$$
在VDM中，ELBO被分解成了三项：

- $E[log(p_\theta (x_0|x_1))] $ ：重建项。给定第一步的隐变量 $x_1$ ，预测原始数据 $x_0$ 。
- $E[KL(q(x_T|x_{T-1})||p(x_T))] $ ：先验匹配项。让最后一步的隐变量的分布匹配正态分布的先验。该项不含任何需要优化的参数，当 $T$ 足够大时，该项自然是0
- $E[KL( q(x_t|x_{t-1})  || p_\theta (x_t | x_{t+1}))]$ ：一致性项。确保加噪和去噪的分布一致性，即从噪声图像去噪时的分布需要与从干净图像加噪时的分布对应。可以通过训练 $p_\theta (x_t | x_{t+1})$ 去匹配 $q(x_t | x_{t-1})$ 来最小化该项。

优化VDM的开销主要在第三项上，因为需要优化所有的时间步 $t$ 。**此外，第三项需要用 $\{ x_{t-1}, x_{t+1} \}$ 两个随机变量来计算期望，其蒙特卡洛估计可能会比只用一个随即变量计算具有更大的方差，这是由于需要对 $T-1$ 个一致性项求和，当 $T$ 较大时，估计的最终的 ELBO 的方差可能也会更大。**

为了让每一项的期望计算都只使用一个随机变量，需要用到关键的公式（马尔可夫链中，$x_t$ 只依赖于 $x_{t-1}$ ，而与 $x_0$ 无关 `where the extra conditioning term is superfluous due to the Markov property --- 大一统论文`）：
$$
q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0)

\\=

\frac{q(x_{t-1}|x_t, x_0) q(x_t|x_0)}{q(x_{t-1}|x_0)}
$$
因此，ELBO可以重写：
$$
E[log(\frac{p(x, z_{1:T}) }{q_\phi (z_{1:T}|x)})]

\\=

E[log(\frac{p(x_{0:T}) }{q (x_{1:T}|x_0)})]

\\=

E[log( \frac{p(x_T) \prod_{t=1}^T p_\theta (x_{t-1}| x_t) }{\prod_{t=1}^T q(x_t|x_{t-1})})]

\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1) \prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{q(x_1|x_0)\prod_{t=2}^{T} q(x_t|x_{t-1})})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) + log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) 
+ 
log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{\frac{q(x_{t-1}|x_t, x_0) q(x_t|x_0)}{q(x_{t-1}|x_0)}})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) 
+
log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})
+
log(\prod_{t=2}^{T}\frac{q(x_{t-1}|x_0) }{q(x_t|x_0)})]
$$
其中最后一项：
$$
log(\prod_{t=2}^{T}\frac{q(x_{t-1}|x_0) }{q(x_t|x_0)})]
\\=
log(
\frac{q(x_{1}|x_0) }{q(x_2|x_0)}
\frac{q(x_{2}|x_0) }{q(x_3|x_0)}
...
\frac{q(x_{T-1}|x_0) }{q(x_T|x_0)}
)
\\=
log(\frac{q(x_{1}|x_0) }{q(x_T|x_0)})]
$$
因此：
$$
ELBO =
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) 
+
log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})
+
log(\frac{q(x_{1}|x_0) }{q(x_T|x_0)})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_T|x_0)}) 
+
\sum_{t=2}^{T} log(\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})]
\\=
E[(log(p_\theta(x_0|x_1)))]
+
E[log(\frac{p(x_T)}{q(x_T|x_0)})]
+
\sum_{t=2}^{T} E[log(\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})]
\\=
E[(log(p_\theta(x_0|x_1)))]
-
KL(q(x_T|x_0)||p(x_T))
-
\sum_{t=2}^{T} E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]
$$
至此，成功的推导了ELBO的一种解，并且可以用较低的方差进行估计，因为每一项都最多只计算一个随机变量的期望。这个公式该有一个优雅的解释：

- $E[(log(p_\theta(x_0|x_1)))]$ ：重构项，该项可以**使用蒙特卡洛估计进行近似和优化**。
- $KL(q(x_T|x_0)||p(x_T))$ ：表示最终的噪声与标准高斯先验的接近程度。该项没有可学习参数，并且基于假设，该项总有 $x_T \in N(0, I)$ ，即该项等于0
- $\sum_{t=2}^{T} E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]$ ：去噪匹配项。我们学习去噪过程 $p_\theta (x_{t-1} | x_t)$ 作为真实去噪过程 $q(x_{t-1}|x_t, x_0)$ 的近似值。

在优化推导的ELBO时存在的主要问题：

- $\sum_{t=2}^{T} E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]$ 项具有 $T - 1$ 个时间步的概率分布需要同时学习，学习难度增加
- 对于任意时刻的 $x_t$ ，都需要从 $x_0$ 逐步加噪获得，复杂度增加
- $p_\theta(x_{t-1}|x_t))$ 是网络学习得到的，但是 $q(x_{t-1}|x_t, x_0)$ 却是未知的

为解决上述问题，可以利用高斯转移假设来优化问题：

根据贝叶斯定理：
$$
q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
$$
并且，由于不同时刻的隐变量都是线性高斯模型，因此：
$$
x_t = 
\sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1}} x_{0} 
+ 
\sqrt{1 - \alpha_t\alpha_{t-1} \alpha_{t-2}...\alpha_{1}} \bar{\epsilon}_{0}

\\
=
\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}_{0}

\sim N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)
$$
即，此时可以直接从 $x_0$ 获取任意时刻的隐变量 $x_t$ 。

此外，从 $x_{t-1}$ 获取 $x_t$ 的方式为 :
$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} 
\sim N(\sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
$$
所以，
$$
q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
\\ \sim \frac{
N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_{0} , 1 - \bar{\alpha}_{t-1}) 
N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
}
{N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)}

\\ \propto
exp\{
-[

\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{2 (1 - \alpha_t)}
+
\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{2 (1 - \bar{\alpha}_{t-1})}
+
\frac{(x_t - \sqrt{\bar{\alpha}_t} x_{0})^2}{2(1 - \bar{\alpha}_t)}
]
\}

\\=

exp
\{
-\frac{1}{2}
[
\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{(1 - \alpha_t)}
+
\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{(1 - \bar{\alpha}_{t-1})}
+
\frac{(x_t - \sqrt{\bar{\alpha}_t} x_{0})^2}{(1 - \bar{\alpha}_t)}
]
\}

\\=
exp
\{
-\frac{1}{2}
[
\frac{\alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1 - \alpha_t)}
+
\frac{x_{t-1}^2 -2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1}}{(1 - \bar{\alpha}_{t-1})}
+
C(x_t, x_0)
]
\}

\\ \propto
exp
\{
-\frac{1}{2}
[
\frac{\alpha_tx_{t-1}^2}{(1 - \alpha_t)}
-
\frac{2\sqrt{\alpha_t}x_tx_{t-1}}{(1 - \alpha_t)}
+
\frac{x_{t-1}^2}{(1 - \bar{\alpha}_{t-1})}
-
\frac{2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1}}{(1 - \bar{\alpha}_{t-1})}
]
\}

\\=
exp
\{
-\frac{1}{2}
[
(\frac{\alpha_t}{(1 - \alpha_t)} + \frac{1}{(1 - \bar{\alpha}_{t-1})})x_{t-1}^2
-2
(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
]
\}

\\=
exp
\{
-\frac{1}{2}
[
(\frac{
\alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)
}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})})x_{t-1}^2
-2
(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
]
\}

\\=
exp
\{
-\frac{1}{2}
[
(
\frac
{\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
)

x_{t-1}^2
-2
(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
]
\}

\\=
exp
\{
-\frac{1}{2}
[
(
\frac
{1 - \bar{\alpha}_{t}}
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
)

x_{t-1}^2
-2
(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
]
\}

\\=
exp
\{
-\frac{1}{2}

(
\frac
{1 - \bar{\alpha}_{t}}
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
)
[
x_{t-1}^2
-2
\frac
{(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})}
{\frac
{1 - \bar{\alpha}_{t}}
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}}

x_{t-1}
]
\}

\\=
exp
\{
-\frac{1}{2}

(
\frac
{1 - \bar{\alpha}_{t}}
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
)
[
x_{t-1}^2
-2
\frac
{(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})}) (1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}

x_{t-1}
]
\}

\\=
exp
\{
-\frac{1}{2}

(
\frac
{1 - \bar{\alpha}_{t}}
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
)
[
x_{t-1}^2
-2
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}

x_{t-1}
]
\}
$$
其中：

- $C(x_t, x_0)$ 是关于 $x_0, x_t$ 的常数项。
- 通过最终的推导形式，可以证明 $x_{t-1} \in q(x_{t-1}| x_t, x_0)$ 是一个高斯分布。

- 对照高斯分布的标准形式：
  $$
  Exp(- \frac{(x - \mu)^2}{2\sigma^2}) = Exp(-\frac{1}{2} \frac{x^2 + \mu^2 -2x\mu}{\sigma^2})
  \\=
  Exp(
  -\frac{1}{2} 
  (
  \frac{1}{\sigma^2} x^2
  -
  \frac{2\mu}{\sigma^2} x
  +
  \frac{\mu^2}{\sigma^2}
  )
  )
  
  \\=
  Exp(
  -\frac{1}{2} 
  (
  \frac{1}{\sigma^2}
  [
   x^2
   -
   2\mu x
  ]
  +C
  )
  $$
  可知：
  $$
  \sigma^2 =
  \frac
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  {1 - \bar{\alpha}_{t}}
  
  \\ 
  \mu =
  \frac
  {
  
  \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
  + 
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
  
  }
  {1 - \bar{\alpha}_{t}}
  $$

即：
$$
q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) 
\sim
N(
x_{t-1};
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}},

\frac
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}
)
$$
可以发现:

- $q(x_{t-1}|x_t, x_0)$ 的均值与 $t, x_t, x_0$ 有关
- 方差只与 $\alpha_t$ 有关，而 $\alpha_t$ 要么在每个时间步都是固定的，要么是作为一组超参数，要么是提前学习好的网络对当前时间步的推理输出，总之与时间 $t$ 有关。
- 因此，为了后续表示的简单性，把均值方差分别表示为关于 $x_t, x_0$ 或 $t$ 的函数，即  $\mu_q(x_t, x_0),  \sigma_q(t)$

接下来为了让网络预测的 $p_\theta (x_{t-1}| x_t)$ 匹配 gt $q(x_{t-1}|x_t, x_0)$ ：

- 把 $p_\theta (x_{t-1}, x_t)$ 也建模成高斯分布，均值为 $\mu_\theta, \sigma_\theta$
- 由于gt $q(x_{t-1}|x_t, x_0)$ 的方差只与 $t$ 有关，不随 $x_0, x_t$ 变化，因此在对齐预测值和真实值时，不用考虑对齐方差。 
- 由于 $p_\theta (x_{t-1}| x_t)$  的条件只有 $x_t$ 而没有 $x_0$ ，因此其均值只是 $x_t$ 的函数，表示成 $\mu_\theta (x_t, t)$ 

因此，根据多元高斯分布的KL散度公式（具体过程略，详见论文中公式87 - 公式92），可以得到 $$KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))$$ 的化简形式：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
$$
即，我们只需要优化两个均值即可。gt的均值刚才推导过，但是预测值的均值未知，可以对照gt的均值，得到：
$$
\mu_q(x_t, x_0) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}
$$

$$
\mu_\theta(x_t, t) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{x_\theta}(x_t, t)

}
{1 - \bar{\alpha}_{t}}
$$

其中，$\hat{x_\theta}(x_t, t)$ 是我们通过噪声输入，来预测出的原始图像。

因此，最小化KL散度可以表示为：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{x_\theta}(x_t, t)

}
{1 - \bar{\alpha}_{t}}


-


\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{x_\theta}(x_t, t)

}
{1 - \bar{\alpha}_{t}}


-


\frac
{
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) 

}
{1 - \bar{\alpha}_{t}}
(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
最终，VDM 的优化目标可以表示为：以任意时刻的噪声输入 $x_t$ ，预测原始图像 $\hat{x_0}$ ，并计算其与真实输入的 $x_0$ 的误差。

## 4.1 Three Equivalent Interpretations

### 4.1.1 预测 $x_0$

如上推理，训练一个扩散模型可以简单的通过从 $x_t$ 预测原始自然图像 $x_0$ 进行学习。除此之外，还有两种等价形式。

### 4.1.2 预测噪声

根据：
$$
x_t =
\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
可以得到：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
$$
把 $x_0$ 带入到 $\mu_q(x_t, x_0)$ ：
$$
\mu_q(x_t, x_0) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}

}
{1 - \bar{\alpha}_{t}}


\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
(1 - \alpha_t) \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\alpha}_t}

}
{1 - \bar{\alpha}_{t}}


\\=

\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}} 
x_t

+

\frac
{1 - \alpha_t}
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t} 
x_t
-

\frac
{ (1 - \alpha_t) \sqrt{1 - \bar{\alpha}_t} }
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t}
\bar{z}_{0}

\\=

(
\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}} 


+

\frac
{1 - \alpha_t}
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t} 
)
x_t
-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}


\\=

(
\frac
{\alpha_t (1 - \bar{\alpha}_{t-1})}
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t})} 


+

\frac
{1 - \alpha_t}
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t} 
)
x_t
-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}

\\=


\frac
{
\alpha_t (1 - \bar{\alpha}_{t-1})
+
1 - \alpha_t
}
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t})} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}


\\=


\frac
{ 1 -\bar{\alpha}_{t} }
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t})} 
x_t
-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}

\\=


\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}
$$
同理，我们可以把模型预测的 $\mu_\theta (x_t, t)$ 表示为：
$$
\mu_\theta (x_t, t) =

\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{\theta}(x_t, t)
$$
这种情况下，模型优化的目标为：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [
||

(
\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{\theta}(x_t, t)
)

-

(\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0})
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [
||


\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{\theta}(x_t, t)

||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [
||


\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
(\bar{z}_{0} - \bar{z}_{\theta}(x_t, t))

||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)}
\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t} [
||



(\bar{z}_{0} - \bar{z}_{\theta}(x_t, t))

||^2_2]
$$
其中，$\bar{z}_{\theta}(x_t, t)$ 是神经网络预测的噪声 $\bar{z}_0 \sim N(0, 1)$ ，表示从 $x_0$ 一步到 $x_t$ 的噪声。上述过程证明了，模型预测原始的图像 $x_0$ 与预测噪声是完全等价的。

此外，一些工作发现，预测噪声的效果会更好。

### 4.1.3 Score-based

由于
$$
x_t = 

\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}

\sim N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)
$$
根据 Tweedie's Formula 可以得到： 
$$
E[\mu_{x_t} | x_t] = x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t)) = \sqrt{\bar{\alpha}_t} x_{0}
$$
为了简单起见，后续把 $\nabla_{x_t} log(p(x_t))$ 简写成 $\nabla log(p(x_t))$

因此，
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
之前推导过：
$$
\mu_q(x_t, x_0) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}
$$
把 $x_0$ 带入得到：
$$
\mu_q(x_t, x_0) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
(1 - \alpha_t) \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\alpha}_t}

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t}
{1 - \bar{\alpha}_{t}}

+

\frac
{(1 - \alpha_t) x_t}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}

+

\frac
{(1 - \alpha_t) (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}

\\=

(
\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
)
x_t


+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{\alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{\alpha_t  - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{1 - \bar{\alpha}_{t}}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{1}
{\sqrt{\alpha}_t}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))
$$
但是，实际数据分布 $p(x_t)$ 是不可能知道的，因此使用神经网络来进行预测 $s_\theta(x_t, t)  \approx  \nabla_{x_t} log(p(x_t))$ 。即：
$$
\mu_q(x_t, x_0) =

\frac
{1}
{\sqrt{\alpha}_t}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
s_\theta(x_t, t)
$$
因此，最终的优化问题为：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{1}
{\sqrt{\alpha}_t}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
s_\theta(x_t, t)

-

\frac
{1}
{\sqrt{\alpha}_t}
x_t

-

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
s_\theta(x_t, t)

-

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} 
\frac
{(1 - \alpha_t)^2}
{\alpha_t}
[||

s_\theta(x_t, t)

-
\nabla_{x_t} log(p(x_t))
||^2_2]
$$
然而，$\nabla_{x_t} log(p(x_t))$ 是并不知道的。回过头再看 $x_0$ 的表达式：
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
其中，$x_t$ 是初始噪声， $\nabla_{x_t} log(p(x_t))$ 是对数似然的梯度方向，对应函数增长最快的方向，沿着梯度方向，就可以使似然函数增大。按照这个观点， $(1 - \bar{\alpha}_t)$ 就可以类比为步长/学习率。

即：从随机初始化的高斯噪声 $x_t$ 开始，沿着 $x_t$ 的对数似然的梯度方向进行更新，使 $x_t$ 的对数似然不断增大，就可以获得真实的数据分布 $p(x)$ 。这个过程也等价于从随机噪声 $x_t$ 开始，每步逐渐去噪。该流程也可以用数学定义严格证明：

由于:
$$
x_t = 

\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
所以：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
$$
对比基于对数似然的梯度的表达式：
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
可以得出：
$$
(1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t)) = - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
即：
$$
\nabla_{x_t} log(p(x_t)) = -  \frac{\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{(1 - \bar{\alpha}_t) }
$$
即：

$x_t$ 沿着对数似然的梯度方向更新，对应于按照一定的比例减去 $\frac{\sqrt{1 - \bar{\alpha}_t} }{(1 - \bar{\alpha}_t) }$ 噪声。

-----



# 5 Learning Diffusion Noise Parameters

$\alpha_t$ 之前的介绍中，一直是作为提前预设的常量来使用，在DDPM中也是从 $[0.0001, 0.02]$ 线性增加的。然而，$\alpha_t$ 也可以作为一个可学习的参数。如，使用参数为 $\eta$ 的模型，预测第 $t$ 个时间步的 $\alpha_t$ 可以表示为 $\hat{\alpha}_\eta (t)$ 。 

为了优化参数 $\eta$ ，仍然从 $argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))$ 推导：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
\\=

argmin_\theta \frac{1}{2\frac
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}\frac{1 - \bar{\alpha}_{t}}{
(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} (1 - \alpha_t)}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} - \bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
噪声参数 $\alpha_t$ 需要控制随着 $t$  的增加，图像的噪声比例越来越大。本文使用 $SNR = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ 信噪比来计算噪声比例。因此，论文中在推导时，为了构造 $SNR$ 的形式，对上式补了一些项，上式可以继续推导：
$$
=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} - \bar{\alpha}_{t-1}\bar{\alpha}_{t} + \bar{\alpha}_{t-1}\bar{\alpha}_{t} - \bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} (1 - \bar{\alpha}_{t}) - \bar{\alpha}_{t}(1 - \bar{\alpha}_{t-1})}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]


\\=

argmin_\theta \frac{1}{2}
(
\frac
{\bar{\alpha}_{t-1} (1 - \bar{\alpha}_{t})}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 
-
\frac
{\bar{\alpha}_{t}(1 - \bar{\alpha}_{t-1})}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]


\\=

argmin_\theta \frac{1}{2}
(
\frac
{\bar{\alpha}_{t-1}}
{(1 - \bar{\alpha}_{t-1})} 
-
\frac
{\bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t})} 
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
刚才描述的 $SNR = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ ，实际上是由 $SNR$ 的定义 $SNR = \frac{\mu^2}{\sigma^2}$ ，以及 $q(x_t|x_0) \sim N(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t))$ 得到的：
$$
SNR = \frac{\mu^2}{\sigma^2}
\\=

\frac{\bar{\alpha}_tx_0^2}{1 - \bar{\alpha}_t}
$$
其中，$x_0^2$ 由于在输入图像确定时，后续加噪的过程中都不变，因此可以忽略。

最终，
$$
\frac{1}{2}
(
\frac
{\bar{\alpha}_{t-1}}
{(1 - \bar{\alpha}_{t-1})} 
-
\frac
{\bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t})} 
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

\frac{1}{2}
(
SNR(t-1)
-
SNR(t)
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
根据SNR的含义：

- 更高的SNR表示更多原始信号
- 更低的SNR表示更多的噪声
- 这里不是以dB为单位，即没有加log，因此SNR一定为正。

因此，随着 $t$ 的增加， $SNR$ 需要单调递减。定义 $w_\eta(t)$ 是一个神经网络，参数是 $\eta$，且输出是单调递增的，则 $-w_\eta(t)$ 的输出是单调递减的。为了保证 SNR 非负，使用 exp 进行映射：
$$
SNR(t) = exp(-w_\eta (t))
$$
即，
$$
\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t} = exp(-w_\eta (t))
$$
可以得到：
$$
\alpha_t = \frac{exp(-w_\eta(t))}{1+exp(-w_\eta(t))}
$$
由于 exp 一定为正数，因此 $0 \lt \frac{exp(-w_\eta(t))}{1+exp(-w_\eta(t))} \lt 1$ 。因此，$\alpha_t$ 可以使用 Sigmoid 来限制范围，即：
$$
\bar{\alpha_t} = sigmoid(-w_\eta(t))
\\
1 - \bar{\alpha}_t  = sigmoid(w_\eta(t))
$$

 

# Score-based Generative Models

- 太卡了，另开一篇
