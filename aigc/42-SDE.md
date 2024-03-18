# Score-Based Generative Modeling through Stochastic Differential Equations

# 0 引用

- 论文：https://arxiv.org/abs/2011.13456

# 1 介绍

- 引入随机微分方程（Stochastic Differential Equation, SDE），通过缓慢的注入噪声，平滑的把复杂的数据分布转变成已知的先验分布。
- 一个与注入噪声逆过程（reverse-time）的SDE，通过缓慢的移除噪声来把先验分布转换回数据分布。reverse-time SDE 只依赖于时间步相关的数据分布的梯度（如，score）。
- 通过利用score-based生成模型的优点，可以准确的用神经网络估计score，并使用SDE的数值求解器来生成样本。

---

有两种成功的概率生成模型涉及到增加噪声来破环训练数据分布，之后通过逆过程来反转这个过程，从而达到生成模型的目的：

- **Score matching with Langevin dynamics (SMLD)** 。在每个噪声尺度估计数据概率密度函数的梯度，之后用朗之万动力学来降低噪声等级。
- **Denoising diffusion probabilistic modeling (DDPM)** 。训练一系列概率模型来反转每一步的噪声。
- 这两种模型都可以称为基于分数的生成模型。

为了利用新的采样方法，并进一步扩展基于分数的生成模型的功能，我们提出了一个统一的框架，使用随机微分方程 SDE 的视角概括了以前的方法：

- 考虑的是根据扩散过程随时间演变的连续分布，而不是用离散的有限几个噪声分布扰动数据。
- 通过反转该过程，可以把随机噪声转化成数据，实现生成样本的目的。这个反向过程满足逆向SDE，可以从正向SDE导出。
- 主要流程如下图所示：

![image-20231218192633604](./imgs/42-SDE/image-20231218192633604.png)

该方法有三个优点：

- **灵活的采样和似然计算：**
- **可控生成：**
- **统一的架构：**

# 2 背景

## 2.1 Score matching with Langevin dynamics (SMLD)

定义用不同水平的 $\sigma_i$ 加噪后的数据分布为 $p_\sigma(\tilde{x} | x) = N(\tilde{x}; x, \sigma_i^2 I)$ 。其中，$\sigma_i$ 满足 $\sigma_{min} = \sigma_1 < \sigma_2 < ... < \sigma_N = \sigma_{max}$ 。通常 $\sigma_{min}$ 足够小，因此 $p_{\sigma_{min}} \approx p_{data}(x)$ ，并且 $\sigma_{max}$ 足够大，因此 $p_{\sigma_{max}}(x) \approx N(x; 0, \sigma_{max}^2I)$ 。NCSN训练一个神经网络来预测noise-condition的分数网络 $s_\theta(x, \sigma)$ ，目标函数为(单个时间步$t$，所有时间步的loss还要加一个 $\mathbb{E}_t$)：
$$
J(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}} ||^2]
$$
采样是，使用 $M$ 个step的Langevin MCMC从 $p_{\sigma_i}(x)$ 中进行采样：
$$
x_i^m = x_i^{m-1} + \epsilon_i s_{\theta^*} (x_i^{m-1}, \sigma_i) + \sqrt{2\epsilon_i} z_i^m , m = 1,2,...,M
$$
其中， $\epsilon$ 是步长。当 $M \to \infty, \epsilon_i \to 0$ 时，$x_1^M$ 可以认为是从真实数据分布中采样的样本。

## 2.1 补充：从朗之万动力学推导出SDE的标准形式

上帝视角，SDE标准形式：$dx = f(x, t) dt + g(t) dw$ 

朗之万动力学：
$$
x_{t+1} = x_t + \epsilon \nabla_x logp(x_t)+ \sqrt{2 \epsilon} z_i
$$
当 $K \to \infty$ 时，定义 $\Delta t = \epsilon$ ，$\Delta t \to 0$ ：
$$
x_{t+1} - x_t = \Delta t \nabla_x logp(x_t)+ \sqrt{2 \Delta t} z_i
$$
把 $\nabla_x logp(x_t)$ 替换成 $f(x, t)$ ，把 $\sqrt{2}$ 替换成 $g(t)$ , 则：
$$
x_{t + \Delta t} - x_t = f(x, t) \Delta t + g(t) \sqrt{\Delta t} z_i
$$
其中，
$$
\sqrt{\Delta t} z_i \in N(0, \Delta t I)
$$
这里引入布朗运动，则：
$$
w_{t + \Delta t} = w_t + N(0, \Delta t I)
\\=
w_t + \sqrt{\Delta t} z_i
$$
即：
$$
w_{t + \Delta t} - w_t = \sqrt{\Delta t} z_i
$$
代入布朗运动并代入，得到：
$$
x_{t + \Delta t} - x_t = f(x, t) \Delta t + g(t) (w_{t + \Delta t} - w_t)
$$
当 $\Delta t \to 0$ 时：
$$
dx = f(x, t) d t + g(t)dw
$$

## 2.2 Denoising Diffusion Probabilistic Models (DDPM)

 DDPM使用一系列正的噪声系数 $0 < \beta_1, \beta_2, ..., \beta_N < 1$ 。对于每个训练样本$x_0 \in p_{data}(x)$ ，会构造一个离散的马尔科夫链 $\{ x_0, x_1, ..., x_N \}$ ，因此：
$$
p(x_i | x_{i-1}) = N(x_i; \sqrt{1 - \beta_i} x_{i-1}, \beta_i I) 
$$
根据马尔科夫性质及贝叶斯公式：
$$
p_{\alpha_i} (x_i | x_0) = N(x_i; \sqrt{\alpha_i} x_0, (1 - \alpha_i)I) 
$$
其中，$\alpha_i = \prod_{j=1}^i (1 - \beta_j)$ 。

# 3 Score-based Generative modeling with SDEs

上述两种方法成功的关键是用多个噪声尺度扰乱数据分布，本文把这个想法进一步扩展到无限数量的噪声尺度，框架如下图所示。

![image-20231218195621237](./imgs/42-SDE/image-20231218195621237.png)

## 3.1 用SDEs扰乱数据

目标是用一系列连续的时间变量 $t \in [0, T]$ 来重建一个扩散过程 $\{ x(t) \}_{t=0}^T$ 。其中，$x(0) \in p_0$ 是独立同分布的观测数据中的样本，$x(T) \in p_T$ 是先验分布（如高斯噪声）。扩散过程可以用 Ito 的SDE形式表示如下：
$$
dx = f(x, t) dt + g(t) dw
$$

- $w$ 是一个标准维纳过程（如，布朗运动）
- $f(\cdot, t)$ 称作 $x(t)$ 的漂移系数，实现 $\mathbb{R}^d \to \mathbb{R}^d$ 的映射。
- $g(\cdot)$ 是实值函数，称作 $x(t)$ 的扩散系数，是一个标量：$\mathbb{R} \to \mathbb{R}$ 。为了简单起见，我们假设扩散系数是一个标量，而不是 $d \times d$ 的矩阵，并且不依赖于 $x$ 
- $dw$ 表示一个很小的高斯白噪声。

为了简单起见，后续定义 $p_t(x)$ 表示 $x(t)$ 的概率密度函数；$p_{st}(x(t)|x(s))$ 表示从 $x(s)$ 到 $x(t)$ 的转移概率分布。其中，$0 \le s \lt t \le T$ 。通常，$p_T$ 中已经不包含任何 $p_0$ 的信息了，只是一个固定均值和方差的分布。

## 3.2 用SDE逆过程生成样本

从 $x(T) \in p_T$ 开始，并进行逆向 SDE，可以最终获得 $x(0) \in p_0$ 。之前的工作证明扩散过程的逆过程仍然是一个扩散过程，逆向 SDE 的表达式为：
$$
dx = [f(x, t) - g(t)^2 \nabla_x logp_t(x)] dt + g(t) d\bar{w}
$$
其中，$\bar{w}$ 是一个标准维纳过程。$dt$ 是一个负无穷小的时间步。一旦知道每个时间步的边缘分布的对数梯度 $\nabla_x logp_t(x)$ 后，就可以用SDE逆向过程生成 $p_0$ 。

## 3.3 估计SDE的Score

SDE逆向过程中只要建模不同噪声等级下的 $\nabla_x logp_t(x)$ 后，就可以采样数据了。用一个分数模型 $s_\theta(x(t), t)$ 表示依赖于时间 $t$ 的分数估计模型。训练目标为：
$$
J(\theta) =  \lambda(t) \mathbb{E}_{t} \mathbb{E}_{x} [|| s(x(t), t) -  \nabla_{x(t)} logp_{0t}(x(t) | x(0)) ||^2_2]
$$
其中：

- $\lambda(t)$ 表示正的加权系数，$t$ 从 $[0, T]$ 中均匀采样。
- $x(0) \in p_0(x)$ ，$x(t) \in p_{0t}(x(t) | x(0))$ 
- $\lambda \propto 1/ \mathbb{E}[|| \nabla_{x(t)} logp_{0t}(x(t) | x(0)) ||_2^2]$

## 3.4 VE, VP, SUB-VP SDEs

主要为了证明SMLD, DDPM都是SDE的一种特殊形式:

- Variance Exploding (VE) ：方差爆炸，对于 SMLD，扩散公式为 $x_T = x_0 + \sigma_T \epsilon$ ，需要加一个特别大的方差，才能让 $X_T$ 变成高斯噪声，因此称作方差爆炸。
- Variance Preserving (VP) ：方差缩紧，对于DDPM，扩散公式为 $x_T = \sqrt{\bar{\alpha}_T} x_0 + \sqrt{1 - \bar{\alpha}_T} \epsilon$ ，主要依靠较小的 $\sqrt{\bar{\alpha}_T}$ 来压制原始图像，并且每一步添加的噪声方差为 $\sqrt{1 - \bar{\alpha}_T}$ 不算大，因此称作方差缩进。

为了证明VE (SMLD) 和 VP(DDPM)都是SDE的一种特殊形式，就要想办法把其扩散公式转化成 $dx = f(x, t) dt + g(t) dw$ 形式。

### 3.4.1 VE SDE (SMLD)

首先还是考虑原始的离散形式下，SMLD加噪公式为：
$$
x_t = x_0 + \sigma_t \epsilon \in N(x_0, \sigma_t^2)
$$

$$
x_{t-1} = x_0 + \sigma_{t-1} \epsilon \in N(x_0, \sigma_{t-1}^2)
$$

想要逐步的向连续靠，就要想办法用 $x_{t-1}$ 表示 $x_t$ ：
$$
x_t \in N(x_0, \sigma_t^2 ) = N(x_0, \sigma_t^2 - \sigma_{t-1}^2 +  \sigma_{t-1}^2)
\\=
x_{t-1} + N(0, \sigma_t^2 - \sigma_{t-1}^2) 
\\=
x_{t-1} + \sqrt{\sigma_t^2 - \sigma_{t-1}^2} \epsilon
$$
其中，$t = 1,2,...,N$ 。当 $N \to \infty$ 时，马尔科夫链 $\{ x_t \}_{t=1}^N$ 可以近似看成是连续的 $\{ x(t) \}_{t=0}^1$。如果把上式看作连续的，符号可以稍微改一下：
$$
x_{t + \Delta t} = x_{t} + \sqrt{\sigma_{t+\Delta t}^2 - \sigma_{t}^2} \epsilon
\\=
x_{t} + \sqrt{\frac{\sigma_{t+\Delta t}^2 - \sigma_{t}^2}{\Delta t}} \sqrt{\Delta t} \epsilon
\\=
x_{t} + \sqrt{\frac{\Delta \sigma_t^2}{\Delta t}} \sqrt{\Delta t} \epsilon
$$
当 $\Delta t \to 0$ 时，且用离散的符号替换，有：
$$
x(t+\Delta t) - x(t) = \sqrt{\frac{\Delta \sigma(t)^2}{\Delta t}} \sqrt{\Delta t} \epsilon
\\
dx = \sqrt{\frac{d [\sigma(t)^2]}{d t}} \sqrt{\Delta t} \epsilon
\\=
\sqrt{\frac{d [\sigma(t)^2]}{d t}} d \epsilon
$$
其中，$t \in [0, 1]$ ，而不再是 $1, 2, ..., i, ..., N$ ，因此 $t = i / N$ 。

对照SDE标准形式 $dx = f(x, t) dt + g(t) dw$ ，可以确定：

- $f(x_t, t) = 0$
- $g(t) = \sqrt{\frac{d [\sigma(t)^2]}{d t}}$
- $w = \epsilon \in N(0, I)$ 

### 3.4.2 VP SDE (DDPM)

和SMLD类似，还是先考虑离散情况：
$$
x_i = \sqrt{1 - \beta_i} x_{i-1} + \sqrt{\beta_i} z_{i-1}
$$
其中，$i = 1, 2, ..., N$ 。

个人理解，为了凑成SDE的标准形式，所以论文里定一个辅助变量，令 $\{ \bar{\beta}_i = N \beta_i \}_{i=1}^N$ 。因此上式可以表示成：
$$
x_i = \sqrt{1 - \frac{\bar{\beta}_i}{N}} x_{i-1} + \sqrt{\frac{\bar{\beta}_i}{N}} z_{i-1}, i=1,2,...,N
$$
并且，当 $N \to \infty$ 时，定义连续的 $\beta(t)$ 是辅助变量，即：
$$
\beta(t) = \{ \bar{\beta}_i \}_{i=1}^N , t \in [0, 1]
$$
 因此，

-  $t = \frac{i}{N}$ 

- $\beta(\frac{i}{N}) = \bar{\beta}_i$ 
- $x(\frac{i}{N}) = x_i$
- $z(\frac{i}{N}) = z_i$ 
- $\Delta t = \frac{1}{N}$ ，$t \in \{ 0, 1, ..., \frac{N - 1}{N} \}$ 

所以上式可以进一步写成：
$$
x(t + \Delta t) = \sqrt{1 - \frac{\bar{\beta}_i}{N}} x(t) + \sqrt{\frac{\bar{\beta}_i}{N}} z(t)
\\=
\sqrt{1 - \frac{\beta(t + \Delta t)}{T}} x(t) + \sqrt{\frac{\beta(t + \Delta t)}{T}} z(t)
\\=
\sqrt{1 - \beta(t + \Delta t) \Delta t} x(t) + \sqrt{\beta(t + \Delta t) \Delta t} z(t)
$$
由于当 $x \to 0$ 时，$(1 - x)^{\alpha} \approx 1 - \alpha x$ ，所以当 $\Delta t \to 0$ 时，上式可以近似写成：
$$
x(t + \Delta t) = \sqrt{1 - \beta(t + \Delta t) \Delta t} x(t) + \sqrt{\beta(t + \Delta t) \Delta t} z(t)
\\ \approx 
(1 - \frac{1}{2} \beta(t ) \Delta t) x(t) + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$
 对照标准形式 $dx = f(x, t) dt + g(t)dw$ ：
$$
x(t + \Delta t) 
\approx 
(1 - \frac{1}{2} \beta(t ) \Delta t) x(t) + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$

$$
x(t + \Delta t) - x(t)
\approx 
- \frac{1}{2} \beta(t ) \Delta t x(t) + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$

$$
dx
\approx 
- \frac{1}{2} \beta(t ) x(t) dt  + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$

因此：

- $f(x, t) = - \frac{1}{2} \beta(t ) x(t)$
- $g(t) = \sqrt{\beta(t) }$

---

另外，大一统中也证明过DDPM和score function的等价形式。

### 3.4.3 VP SDEs

没看懂，是基于DDPM改进的一种，表现更好，前向 SDE 表达式为：
$$
dx
=
- \frac{1}{2} \beta(t ) x(t) dt  + \sqrt{\beta(t) e^{-2 \int_0^t \beta(s) ds} } dw
$$

# 4 求解SDE

训练好 $s_\theta$ 之后，可以使用逆向SDE来求解数值解，从而生成 $p_0$ 样本。

## 4.1 通用数值 SDE 求解器

数值求解器提供了SDE的轨迹的估计，文中介绍了两种通用的SDE数值求解方法：

- Euler–Maruyama method: 欧拉丸山方法，是欧拉方法对于随机微分方程的推广。
- Stochastic Runge–Kutta method: 随机龙格库塔方法，是龙格库塔方法对于随机微分方程的推广。

另外，作者证明了DDPM的采样方法也是VP SDE求解器的一种特殊离散化形式，证明如下。

---

 DDPM 在标准情况下的前向SDE过程为：
$$
dx = 
- \frac{1}{2} \beta(t ) x(t) dt  + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$
所以扩散过程为：
$$
x(t + \Delta t) = x(t) + dx = x(t) + f(x, t) dt + g(t) \sqrt{\Delta t} z(t)
$$
其中，$z(t) \in N(0, I)$ ，因此 $g(t) \sqrt{\Delta t} z(t) \in N(0, g(t)^2 \Delta t)I$ 。

所以，扩散过程的条件概率分布为
$$
p(x_{t+\Delta t} | x_t) = N(x_{t+\Delta t}; x_t + f(x, t) \Delta t, g(t)^2 \Delta t I) 
\\ 
\propto
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t})
$$
为了过的逆向SDE的条件概率分布 $p(x(t) | x(t + \Delta t))$ ，根据贝叶斯公式：
$$
p(x(t) | x(t + \Delta t)) = \frac{p(x_{t + \Delta t} | x_t) p(x_t)}{p(x_{t + \Delta t})}
\\=
p(x_{t + \Delta t}) exp(logp(x_t) - logp(x_{t+\Delta t}))

\\ \propto
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} + logp(x_t) - logp(x_{t + \Delta t}))
$$
其中，对 $logp(x_{t + \Delta t})$ 做泰勒展开（$p(x_t)$ 是双变量 $x_t, t$ 的函数，所以展开式有两项）：
$$
logp(x_{t + \Delta t}) \approx logp(x_t) + (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) + \Delta t \frac{\partial logp(x_t)}{\partial t}
$$
把泰勒展开代入：
$$
p(x(t) | x(t + \Delta t))  \propto
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} + logp(x_t) - logp(x_{t + \Delta t}))
\\=
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} + logp(x_t) - logp(x_t) - (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) - \Delta t \frac{\partial logp(x_t)}{\partial t}

\\=
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} - (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) - \Delta t \frac{\partial logp(x_t)}{\partial t}
$$
由于 $\Delta t \to 0$ ，所以上式中最后一项 $\to 0$ ，即：
$$
p(x(t) | x(t + \Delta t))  \propto
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} - (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) - \Delta t \frac{\partial logp(x_t)}{\partial t}

\\ \approx
exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} - \frac{2g(t)^2 \Delta t(x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t)}{2g(t)^2 \Delta t})

\\=
exp(- \frac{|| x_{t + \Delta t} - x_t - [f(x, t) - g(t)^2 \nabla_{x_t} logp(x_t) ] \Delta t ||^2}{2g(t)^2 \Delta t})
$$
由于当 $\Delta t \to 0$ 时，$t + \Delta t \to t$ 。同时由于逆向过程中，$t + \Delta t$ 是已知的，为 $t$ 未知，因此用 $t + \Delta t$ 替换 $t$ ：
$$
p(x(t) | x(t + \Delta t)) \approx
exp(- \frac{|| x_{t + \Delta t} - x_t - [f(x, t) - g(t)^2 \nabla_{x_t} logp(x_t) ] \Delta t ||^2}{2g(t)^2 \Delta t})

\\=

exp(- \frac{|| x_{t} - x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] \Delta t ||^2}{2g(t + \Delta t)^2 \Delta t})
$$
即：
$$
p(x(t) | x(t + \Delta t))  \in N(x(t); x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] \Delta t, g(t + \Delta t)^2 \Delta t I)
$$
由于逆向SDE中，$dx = x_{t + \Delta t} - x_t$ ，并且其中的 $x_t$ 可以根据 $p(x(t) | x(t + \Delta t))$ 进行采样：
$$
x_t = x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] \Delta t + g(t + \Delta t) \sqrt{\Delta t} w
\\=
x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] dt + g(t + \Delta t) dw
$$
因此：
$$
dx = x_{t + \Delta t} - x_t 
\\=
[f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] dt - g(t + \Delta t) dw
$$
再取 $\Delta t \to 0$ ：
$$
dx = x_{t + \Delta t} - x_t 
\\=
[f(x_{t}, t) - g(t)^2 \nabla_{x_{t}} logp(x_{t}) ] dt - g(t) dw
$$

---

标准的SDE逆向过程是：
$$
dx = [f(x_{t}, t) - g(t)^2 \nabla_{x_{t}} logp(x_{t}) ] dt + g(t) dw
$$
正负号应该对于采样高斯噪声没有影响。

## 4.2 DDPM去噪过程等价于SDE数值解求解器

DDPM离散过程的去噪公式为：
$$
x_{t-1} = 

\frac{1}{\sqrt{\alpha_{t}}}
(x_t - \frac
{(1-\alpha_t)}
{\sqrt{1 -  \bar{\alpha}_{t}}}
\bar{z}_{0})

+

\sigma z
$$
大一统中根据 Tweedie's Formula得到：
$$
\nabla_{x_t} log(p(x_t)) = -  \frac{\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{(1 - \bar{\alpha}_t) } = \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \bar{z}_{0}
$$
即：
$$
x_{t-1} = 

\frac{1}{\sqrt{\alpha_{t}}}
(x_t + (1 - \alpha_t) \nabla_{x_t} log(p(x_t)))

+

\sigma z
$$
=== $\sigma$ 和论文附录中的对不上，反复检查没有看出问题在哪：

- 论文中 $\sigma^2 = \beta_t = 1 - (\alpha_t)$
- DDPM中，$\sigma^2 = \frac{(1-\alpha_t) (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}$
- 但是没有关键影响，这里就用 $\sigma$ ，不带入具体表达式

---

- 按照论文，用 $\beta = 1 - \alpha$ 代替。

- 需要用到极限：$lim_{x \to 0} (1-x)^{a} \approx 1 - a x$

---

DDPM逆向过程继续展开：
$$
x_{t-1} = 

\frac{1}{\sqrt{1 - \beta_t}}
(x_t + \beta_t \nabla_{x_t} log(p(x_t)))

+

\sigma z

\\ \approx

(1 + \frac{1}{2} \beta_{t})
(x_t + \beta_t \nabla_{x_t} log(p(x_t)))
+
\sigma z

\\ =

(1 + \frac{1}{2} \beta_{t}) x_t
+
(1 + \frac{1}{2} \beta_{t}) \beta_t \nabla_{x_t} log(p(x_t))
+
\sigma z

\\ =

(1 + \frac{1}{2} \beta_{t}) x_t
+
\beta_t \nabla_{x_t} log(p(x_t))
+
\frac{1}{2} \beta_{t}^2 \nabla_{x_t} log(p(x_t))
+
\sigma z

\\ \approx

(1 + \frac{1}{2} \beta_{t}) x_t
+
\beta_t \nabla_{x_t} log(p(x_t))
+
\sigma z

\\=
[2 - (1 - \frac{1}{2} \beta_{t}] x_t
+
\beta_t \nabla_{x_t} log(p(x_t))
+
\sigma z

\\ \approx
[2 - \sqrt{1 - \beta_{t}}] x_t
+
\beta_t \nabla_{x_t} log(p(x_t))
+
\sigma z
$$
其中，三个约等于中的两个是根据极限得到的，另外一个猜测是根据值比较小因此舍去的。

对照SDE逆向过程标准形式，上式左右两边同时减去 $x_t$ 构造 $\Delta x$ ，并令 $t \to 0$ ，就可以得到类似下式的标准形式：
$$
dx = [f(x_{t}, t) - g(t)^2 \nabla_{x_{t}} logp(x_{t}) ] dt + g(t) dw
$$


## 4.3 Predictor - Corrector (PC sampling)

在没有把DDPM和SMLD统一到SDE之前，其各自也都能求解。统一之后，两种方法都新增了一种利用SDE数值解求解器求解的方法。结合SDE数值解求解器，并结合各自原始的求解方法（如SMLD的朗之万MCMC），可以做到更准确的采样：

- 在每个step，首先由 SDE 求解器估计出下一个step的样本。SDE求解器的角色就是 "Predictor" 。
- 之后，基于score的 MCMC方法来修正样本。角色是 "Corrector" 。

PC的整体算法如下图所示：

![image-20231220181229975](./imgs/42-SDE/image-20231220181229975.png)

以刚推导过的DDPM为例，并用 reverse diffusion SDE 求解器作为 predictor，用退火朗之万动力学作为 corrector，算法如下：

![image-20231220182341394](./imgs/42-SDE/image-20231220182341394.png)

## 4.4 PF-ODE

等价 ODE形式如下，称为 Probability Flow ODE (PF-ODE)：
$$
dx = [f(x, t) - \frac{1}{2} g(t)^2 \nabla_x logp_t(x)] dt
$$
推导略，截图如下：

![image-20231220163332119](./imgs/42-SDE/image-20231220163332119.png)

下图展示了SDE和ODE解的过程，可以看到ODE的轨迹是确定光滑的，而SDE的轨迹是随机的。这两个过程中的任意边缘分布 $\{ p_t(x) \}_{t \in [0, T]}$ 都是一样的。

![img](./imgs/42-SDE/v2-31764a1bf4a3b9f2aa0b4dc2ffbacc91_720w.webp)

ODE形式有它的优点和缺点：

- Pros
  - 因为ODE比SDE好解，所以ODE的采样速度更快
  - 因为ode是不带随机噪声的，整个过程是确定的，也是可逆的，所以这个ode也可以看做normalizing flow，可以用来估计概率密度和似然

- Cons
