# Diffusion Model with Perceptual Loss

# 0 引用

- 论文：https://arxiv.org/pdf/2401.00110.pdf

# 1 介绍

- 逐像素计算MSE损失的方法对于人眼无法察觉到的差异也会有较大的损失，和人类感知无法很好的对齐。

# 2 方法

定义：

- 原始隐空间编码 $x_0 \in \pi_0$
- 随机高斯噪声 $\epsilon \in N(0, I)$ 
- 时间步 $t \in U(1, T)$ 

- 加噪过程定义为 $x_t = forward(x_0, \epsilon, t) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

- 扩散模型 $f_\theta$ 的文本条件 $c$ ，预测 $v$ ：
  $$
  GT : v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} x_0
  \\
  Pred : \hat{v}_t = f_\theta(x_t, t, c)
  $$

- v-prediction计算损失：
  $$
  L_{mse} = || \hat{v}_t - v_t ||_2^2
  $$

## 2.1 Self-Perceptual Objective

为了提供感知信息，通常需要额外的网络提取特征，如CLIP，Imagenet预训练模型，无监督预训练模型等。本文认为扩散模型本身也可以提供有价值的感知损失。

- 对于用 v-prediction 预训练好的模型，拷贝两份，一份作为可微调的在线模型 $f_\theta $ ，另一份冻结权重 $f_*$。

- 采样 $x_0 \in \pi_0$ ，$\epsilon \in N(0, I)$ ，$t \in U(1, T)$ ，加噪得到 $x_t = forward(x_0, \epsilon, t)$ 

  - 使用微调模型预测 $\hat{v}_t = f_\theta(x_t, t, c)$ 
  - 计算出 $\hat{x}_0$ 和 $\hat{\epsilon}$ ：
    - $\hat{x}_0 = \sqrt{\bar{\alpha}}_t x_t - \sqrt{1 - \bar{\alpha}_t} \hat{v}$
    - $\hat{\epsilon} = \sqrt{\bar{\alpha}}_t \hat{v} - \sqrt{1 - \bar{\alpha}_t} x_t$

- 重新采样一个时间步 $t' \in U(1, T)$  ，分别对真实图像 $x_0$ 和 预测图像 $\hat{x_0}$ 进行加噪：

  - $x_{t'} = forward(x_0, \epsilon, t')$
  - $\hat{x}_{t'} = forward(\hat{x}_0, \hat{\epsilon}, t') $

- 目前，有了两个隐变量：

  - 一个是直接从原图加噪的 $x_{t'}$
  - 一个是线上模型预测出的原图，再加噪的 $\hat{x}_{t'}$ 

- 为了保证线上模型预测出的原图  和  真实的原图尽可能相同，通常可以用MSE，但本文在这个地方使用了自感知损失：只对齐感知网络提取的 $x_{t'}$ 和 $\hat{x}_{t'}$ 的中间隐层特征：
  $$
  L_{sp} = || f_{*}^l (\hat{x}_{t'}, t', c) - f_{*}^l(x_{t'}, t', c) ||_2^2
  $$

  - 对齐中间隐层的特征其实还是为了对齐线上模型预测的原图。感知网络的输入一个是真正的原图加噪，一个是预测的原图加噪，感知网络只是需要提取特征，让这两个特征对齐即可，所以对感知网络本身的出图质量并没有要求。

  - 其中，$l$ 是用感知特征提取网络的第 $l$ 层特征计算损失，实验发现使用 midblock的输出的效果最好：

    ![image-20240103110154249](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103110154249.png)

# 3 实验

## 3.1 两次加噪的 $t$ 的选择

- $t$ 和 $t'$ 不能相同，否则就会出现：
  $$
  \hat{x}_{t'} = forward(\hat{x}_0, \hat{\epsilon}, t')  = forward(x_0, \epsilon, t)
  $$

- 对于其他时间步 $t'$ 的选择，做了消融实验，发现均匀采样的效果最好：

  ![image-20240103113021713](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103113021713.png)

## 3.2 距离函数选择

- MAE和MSE的效果类似，最终用MSE

![image-20240103113055944](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103113055944.png)

## 3.3 两次加噪的方式

网络预测的是 $v$ ，可以分解成 $x_0$ 和 $\epsilon$ ，在计算感知损失时，二者都应该包含进去，另外一种加噪形式为：

![image-20240103113344163](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103113344163.png)

和前面介绍的第一种加噪方式进行了对比，第一种加噪方式更好一些：

![image-20240103113413235](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103113413235.png)

## 3.4 感知网络

本文用的是冻结权重的预训练网络作为感知网络，另外一种比较自然的想法是用EMA模型，但是实验发现效果不好：

![image-20240103113614188](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103113614188.png)

# 4 伪代码

![image-20240103113815171](./imgs/48-Diffusion Model with Perceptual Loss/image-20240103113815171.png)