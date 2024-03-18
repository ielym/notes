`Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning`

# 1 动机

当前自监督学习可分为两种方法：

+ 一致性正则
+ 直接生成标签

然而，这些方法都存在错位的预测，当使用这些噪声标签不断训练模型时，会产生噪声累积（Confirmation bias），并且使得模型很难产生新的变化（即拟合噪声数据，而很难重新对已经错误标注的样本重新输出正确的标签）。



# 2 本文方法

## 2.1 定义

+ 有标签的数据集 $D_l = \{(x_i, y_i)\}^{N_l}_{i=1}$ ，$y_i \in \{0, 1\}^C$
+ 无标签的数据集 $D_u = \{x_i\}^{N_u}_{i=1}$ 
+ 其中，$C$ 为类别数量，$N = N_l + N_u$ 表示有标签和无标签的样本数量。
+ 定义 $h_{\theta}(x)$ 为CNN模型，$\theta$ 是模型参数。

## 2.2 获取伪标签

+ Warm-UP 10 个epochs，只使用有标签的数据 $D_l$ 。

+ 之后， 使用两种正则来促进模型收敛：

  + 第一种正则主要为了解决在训练早期，模型还未收敛，此时模型预测的标签大多数都是错的，并且模型为了更小的loss，通常倾向于把所有样本都预测成同一个类。因此在这个阶段，让模型向先验概率分布收敛，比向0 1收敛更容易：
    $$
    R_A = \sum_{c=1}^{C} p_c log(\frac{p_c}{\bar{h_c}})
    $$
    上式是一个KL散度，让模型的预测值 $\bar{h_c}$ 的分布接近先验概率分布 $p_c = \frac{1}{C}$ 。此外，$\bar{h_c}$ 并不是每个样本的预测概率分布，而是一个 mini-batch 的所有样本的平均 softmax 概率分布。

  + 第二种正则是最小化熵：
    $$
    R_H = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} h_{\theta}^{c}(x_i)log(h_{\theta}^{c}(x_i))
    $$
    该项正则是对于每个样本都需要计算。即，使模型预测的不确定性最低（熵减）。

+ 与两种正则同时使用的，还有无标签样本的交叉熵：
  $$
  l^{*} = - \sum_{i = 1}^{N} \tilde{y_i} log(h_{\theta}(x_i))
  $$
  其中，$\tilde{y_i}$ 是伪标签，本文使用 soft pseudo-labels，使用网络所有 batch 预测的 softmax 概率分布 $h_{\theta}(x_i)$ 来更新伪标签 $\tilde{y_i}$ 。

+ 总的 loss 为：
  $$
  l = l^{*} + \lambda_A R_A + \lambda_H R_H
  $$
  其中，两个 $\lambda$ 用于控制两种正则项的贡献程度。

## 2.3 Confirmation bias

`Overfitting to incorrect pseudo-labels predicted by the network is know as confirmation bias.`

本文使用 MixUP 数据增强策略来缓解确认偏差。对于一个样本对 $x_p$ 和 $x_q$ ，对应的标签分别为 $y_p$ 和 $y_q$ ：

+  融合之后的样本为  $x = \delta x_p + (1 - \delta) x_q$
+ 融合后的标签为 $y = \delta y_p + (1 - \delta) y_q $

其中， $\delta \in \{0, 1\}$ ，是服从 beta 分布 $Be(\alpha, \beta)$ 的随机变量，并且 $\alpha = \beta$ 。

但是，通常不直接融合标签，而是融合loss：
$$
l^* = \delta l^*_p + (1-\delta)l^*_q
$$

## 2.4 最终的 Loss

$$
l = N_l \bar{l_l} + N_u \bar{l_u}
$$

其中，前一项是有标签的数据，只计算交叉熵。