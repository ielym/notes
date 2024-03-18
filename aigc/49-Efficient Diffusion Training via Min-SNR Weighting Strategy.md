# Efficient Diffusion Training via Min-SNR Weighting Strategy

# 0 引用

- 论文：https://arxiv.org/pdf/2303.09556.pdf

# 1 介绍

对训练模型训练慢的原因进行了分析，认为主要原因是不同时间步的优化方向冲突导致的。为了解决该问题，提出了 $Min-SNR-\gamma$ 损失加权策略。该策略把不同时间步的优化任务当作一个多任务学习任务。为了平衡不同任务，根据不同任务的学习难度来指定损失权重。这种方法能够有效加快收敛速度，如下图所示：

![image-20240106191945543](imgs/49-Efficient%20Diffusion%20Training%20via%20Min-SNR%20Weighting%20Strategy/image-20240106191945543.png)

# 2 Preliminary

定义训练数据分布 $x_0 \in p(x_0)$ ，前向加噪过程获得一系列的隐变量 $x_1, x_2, ..., x_T$ ：
$$
q(x_t | x_0) = N(x_t; \alpha_t x_0, \sigma_t^2 I) 
$$

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon
$$

其中，$\epsilon \in N(0, I)$  ，噪声方差 $\sigma_t$ 随 $t$ 单调递增。本文使用标准的 VP 扩散过程，即：
$$
\alpha_t = \sqrt{1 - \sigma_t^2}
$$
逆向去噪过程的分布为：
$$
p_\theta(x_{t-1}|x_t) = N(x_{t-1}; \hat{\mu}_\theta(x_t), \hat{\Sigma}_\theta(x_t))
$$
损失为：
$$
L_{simple}^t(\theta) = \mathbb{E}_{x_0, \epsilon} [ || \epsilon - \hat{\epsilon}_\theta(\alpha_t x_0 + \sigma_t \epsilon) ||_2^2 ]
$$
此外，也有用 UNet 预测  $x_0$  的，对应的损失为：
$$
L_{simple}^t(\theta) = \mathbb{E}_{x_0, \epsilon} [ || x_0 - \hat{x}_\theta(\alpha_t x_0 + \sigma_t \epsilon) ||_2^2 ]
$$
此外，还有预测 $v$ 的。尽管这些方法预测的目标不同，但是我们可以通过数学方式证明他们都是等价的，只是在损失的权重上有差异。

# 3 方法

## 3.1 把扩散模型训练当作多任务学习

为了节省参数量，扩散模型通常用一个模型来处理不同的时间步。因此，分析不同时间步的相关性至关重要。

