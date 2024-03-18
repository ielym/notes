# Elucidating the Design Space of Diffusion-Based Generative Models

# 0 引用

- 论文：https://arxiv.org/abs/2206.00364

# 1 介绍

现有扩散模型有着丰富的采样策略，训练方式，噪声参数等，这些理论的推导保证了扩散模型具有坚实的理论基础，然而也导致了每种方法都可能过于紧密耦合，修改其中的任何一个组件都可能破坏整个系统。

因此，本文从实践的角度来重新审视这些模型背后的理论，更多的关注关注训练和采样阶段中的“有形”的算法和对吸纳更，而不是统计过程。本文的工作时在基于score-matching的背景下进行的。

此外，本文评估了不同的采样策略，并分析随机性在采样过程中的有用性。实现了显著减少采样步数，并且改进的采样器可以用在集中广泛使用的扩散模型中。

# 2 在通用框架中表示扩散模型

定义：

- 数据分布 $p_{data}(x)$ ，标准差 $\sigma_{data}$ 
- 加入 $\sigma$ 的高斯噪声的数据分布 $p(x; \sigma)$ ，噪声水平最大为 $\sigma_{max}$ ，且 $\sigma_{max} >> \sigma_{data}$ 。$p(x; \sigma_{max})$  通常与纯高斯噪声没有区别。
- 扩散模型的思想是随机采样出一个噪声图像 $x_0 \in N(0, \sigma_{max}^2I)$ ，并逐渐去噪到 $x_i$ 。去噪过程中的噪声方差 $\sigma_0 = \sigma_{max} > \sigma_1 > ... > \sigma_N = 0$ 。每个噪声等级的数据分布 $x_i \in p(x_i; \sigma_i)$ 。终点 $x_N$ 可以认为服从 $p_{data}(x)$ 的分布。
- Song等人提出了一种SDE形式，能够使用SDE求解器进行求解。SDE求解过程中，每次迭代都会去除噪声并加入新的噪声。Song也给出了一种PF-ODE的形式，整个采样过程中的随机性只来自于初始化噪声图像 $x_0$ 。

## 2.1 ODE形式

Song给出了SDE的形式为：
$$
dx = f(x, t)dt + g(t) d w_t
$$
其中，$w_t$ 是标准维纳过程，$f(\cdot, t) : \mathbb{R}^d \to \mathbb{R}^d$ 和  $g(\cdot) : \mathbb{R} \to \mathbb{R}$ 分别表示漂移系数和扩散系数。两个系数的选择对于不同的VP和VE模型有不同的形式，如：

- VP （DDPM） ：$f(x, t) = - \frac{1}{2} \beta(t ) x(t)$ ，$g(t) = \sqrt{\beta(t) }$ 
- VE （SMLD）：$f(x_t, t) = 0$ ，$g(t) = \sqrt{\frac{d [\sigma(t)^2]}{d t}}$ 

- 其中，$f(x, t)$ 总是可以表示成 $f(x, t) = f(t) x$ 的形式，其中 $f(\cdot) : \mathbb{R} \to \mathbb{R}$ 

因此，SDE也可以等价写成：
$$
dx = f(t)x dt + g(t) d w_t
$$


---

