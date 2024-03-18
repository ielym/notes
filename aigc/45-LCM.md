# Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference

# 0 引用

- 论文：https://arxiv.org/abs/2310.04378

# 1 介绍

- 受一致性模型启发，提出了Latent Consistency Modelss (LCMs) ，提出了一种单阶段引导蒸馏方法，通过求解增强的 PF-ODE 来有效地将预训练的引导扩散模型转换为潜在一致性模型
- 在隐空间直接预测ODE的解
- 除了LCM之外，还提出了 Latent Consistency Fine-tuning (LCF)，一种专为在定制图像数据集上微调 LCM 而定制的新方法

# 2 背景

## 2.1 扩散模型

扩散模型定义一个前向过程，把原始数据分布 $p_{data}(x)$ 转换成预定义的边缘分布 $q_t(x_t)$ ，转换过程中的转换kernel为 $q_{0t}(x_t | x_0) = N(x_t; \alpha(t)x_0, \sigma^2(t)I)$ ，$\alpha(t), \sigma(t)$ 定义了噪声策略。

从连续时间的视角来看，前向过程可以表示为一个SDE，$t \in [0, T]$ ：
$$
d x_t = f(t) x_t dt + g(t) d w_t
$$
其中，$x_0 \in p_{data}(x_0)$ ，$w_t$ 是一个标准布朗运动。

