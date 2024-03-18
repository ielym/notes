# Generative Modeling by Estimating Gradients of the Data Distribution

# 0 引用

- 论文：https://arxiv.org/abs/1907.05600

# 1 介绍

- 提出了一种新的生成模型，` samples are produced via Langevin dynamics using gradients of the data distribution estimated with score matching` 。
- 由于当数据在低维流形上时，梯度可能难以定义且难以估计，因此本文使用不同级别的高斯噪声扰动数据，并联合估计对应的scores（估计不同等级高斯噪声的scores）。
- 采样时，提出一种退火的朗之万动力学。

---

本文利用数据分布的概率密度函数的对数的梯度来构建生成模型，概率密度函数的对数梯度是对数的密度函数增长最快的方向。然后使用朗之万动力学进行采样，朗之万动力学通过从一个随机初始化的样本开始，逐步的移动到score的高密度区域。然而这种方法有两个问题：

- 由于现实世界中的图像数据大都分布比较集中，因此数据分布在低密度区域可能未定义（或 $p(x) \approx 0$ ，则 $log(p(x))$ 没有意义）。
- 由于低密度区域的数据稀缺，导致分数估计的准确性降低，并减慢了朗之万动态采样的混合速度。这是由于初始样本通常会定义在低密度区域，因此这些区域中的不准确的分数估计会对采样过程产生负面影响。

为了解决上述问题：

- 本文提出一种用随机高斯噪声来扰乱数据的方法。添加随机噪声可以保证最终得到的数据分布不会崩溃成低维流形。大的噪声水平会在原始未扰动的数据分布的低密度区域中产生样本，从而改善分数估计。
- 此外，本文方法只需要训练一个单个的score network，并用噪声水平作为条件，就可以估计所有不同level的噪声的分数。
- 之后，提出了一种朗之万动力学的退火版本，初始噪声水平与训练时加入的最大噪声水平相同，并追荐对噪声水平进行退火，知道其小到与原始数据的分布无法区分为止。

# 2 Score-based generative modeling

定义：

- 独立同分布的样本 $\{ x_i \in \mathbb{R}^D \}_{i=1}^{N}$ ，样本来自一个未知的数据分布 $p_{data}(x)$ 
- 定义概率密度函数 $p(x)$ 的分数为 $\nabla_x log(p(x))$ 
- 分数网络用 $s_\theta$ 表示。分数网络是一个神经网络，参数为 $\theta$ ，作用是实现映射 $\mathbb{R}^D \to \mathbb{R}^D$ 。
- 目标是用已经有的样本来学习一个网络模型，并能够从数据分布 $p_{data}(x)$ 中采样新样本。

基于 score-based 生成模型有两个组成部分：

- score mathcing
- Langevin dynamics

## 2.1 Score matching for score estimation

原始score matching用于基于未知数据分布中采样的独立同分布的样本，来学习非归一化的统计模型。本文用score matching用于估计score。基于score matching，我们可以直接训练一个score network $s_\theta(x)$ 来估计未知数据分布的对数梯度 $\nabla_x logp_{data(x)}$ ，而不需要事先知道 $p_{data}(x)$ 或事先训练一个模型来估计 $p_{data}(x)$ 。

在原始score matching中：

- 概率密度函数为 
  $$
  p(x) = \frac{1}{Z(\theta)} e^{f(x)}
  $$

- 通过预测score，即 $p(x)$ 的对数的梯度，来实现消掉 $Z$ 的目的：
  $$
  \nabla_x log(p(x)) 
  = 
  \nabla_x f(x)
  =
  s(\theta;x)
  $$

- 目标函数为：
  $$
  J(\theta) = \frac{1}{2} \int_x p(x) || s(x;\theta) -  \nabla_x log(p_\theta(x))||^2 dx
  \\=
  \frac{1}{2} \mathbb{E}_{x} [|| s(x;\theta) -  \nabla_x log(p_\theta(x))||^2]
  $$

- 由于不知道目标函数中的 $log(p(x))$ ，因此score matching基于一定假设，把目标函数转化成：
  $$
  \mathbb{E}_x [\frac{1}{2}|| s_i(x;\theta) ||^2 + \nabla_x^2 s_i(x;\theta)  ||^2]
  $$

- 由于 $\nabla_x f(x) = s(\theta;x)$ ，所以此时就不再需要 $p(x)$ 了，也不再需要 $Z$ 了，而只需要指数上面的 $f(x)$ 即可。

---

但是，生成任务中，通常即不知道 $p(x)$ ，也不知道 $f(x)$ 。为了解决这个问题，论文中引用了一种 `Denoising score matching` 的方法，基本流程为：

- 强行对原始数据 $x$ 加噪，使得加噪后的数据满足我们自己预定义的数据分布 ，如高斯分布 $q_\sigma(\tilde{x} | x)$。

- 根据边缘分布的定义：
  $$
  q_\sigma(\tilde{x}) = \int_x q_\sigma(\tilde{x}, x) dx = \int_x q_\sigma(\tilde{x}| x)p_{data}(x) dx
  $$

- 之后，就可以在特定的已知分布 $q_\sigma(\tilde{x} | x)$ 上去计算：
  $$
  J(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}} ||^2]
  $$
  其中，由于 $p_{data}(x)$ 分布与预定义的 $\tilde{x}$ 无关，因此偏导可以忽略。
  
- 然而，只有当添加的噪声较小时，才会有 $s_\theta^*(x) = \nabla_x log q_\sigma(x) \approx \nabla_x log p_{data}(x)$

## 2.2 Sampling with Langevin dynamics

朗之万动力学可以仅使用score function $\nabla_x log(p(x))$ 就能实现在概率密度函数 $p(x)$ 中采样，采样公式为：
$$
\tilde{x}_t =  \tilde{x}_{t-1} + \frac{\epsilon}{2} \nabla_x logp(\tilde{x}_{t-1}) + \sqrt{\epsilon} z_t
$$

- $\epsilon > 0$ 是步长
- $z_t \in N(0, I)$ 是标准布朗运动。
- 初始值 $\tilde{x}_0 \in \pi(x)$ ，$\pi$ 是先验分布。
- 当 $\epsilon \to 0, T \to \infty$ 时，在一些正则条件下，$\tilde{x}_T$ 会收敛到服从 $p_{data}(x)$ 的分布。反过来说，当 $\epsilon > 0$ 并且 $T < \infty$ 时，Metropolis-Hastings（MH）更新需要修正拉格朗日动力学的误差，但是这个误差通常可以忽略。本文假设当$\epsilon$ 很小并且 $T$ 较大时，该误差可以忽略不计。

# 3 Score-based生成模型的挑战

## 3.1 The manifold hypothesis (流形假说)

流形假设认为，真实数据大都倾向于分布在低维空间中。也就是说，编码空间（如神经网络的输出特征）维度可能很高很大，单实际上只用其中一部分维度就足以表示数据分布，因此说明数据分布仅分布在整个编码空间中的一部分（低维流形），并没有占满整个编码空间。

在流形假说下，score-based生成模型面临两个关键问题：

- 分数 $\nabla_x logp_{data}(x)$ ，该梯度在低维流形中是未定义的。
- 目标函数 $J(\theta) = \frac{1}{2} \mathbb{E}_{x} [|| s(x;\theta) -  \nabla_x log(p_\theta(x))||^2]$ 的推导是基于样本占满了整个编码空间的前提而推出来的，当数据仅分布在低维流形时，score matching的目标函数结论就可能不适用了。

## 3.2 低数据密度区域

低密度区域中的数据稀缺可能会给score matching的估计，以及基于朗之万动力学的MCMC采样带来困难。

### 3.2.1 不准确的score估计

![image-20231218152109728](./imgs/43-Generative Modeling by Estimating Gradients of the Data Distribution/image-20231218152109728.png)

由于在低密度区域缺少样本，因此学习到的 $s_\theta(x)$ 在低密度区域无法准确估计 $logp_{data}(x)$ 的梯度。如上图所示，颜色较深的地方估计的分数和真实的分数基本一致，但是颜色较浅的地方，特别是对角线附近，箭头的差异就比较大。

### 3.2.2 混合数据分布的Langevin dynamics采样速度慢

当真实数据分布是混合数据分布时，如：
$$
p_{data}(x) = \pi p_1(x) + (1 - \pi) p_2(x)
$$

- 其中 $\pi$ 是两个分布的加权系数。

如果单独看两个分布的分数：

- $\pi p_1(x)$ 的分数：
  $$
  \frac{\partial log(\pi p_1(x))}{\partial x}
  \\=
  \frac{\partial [log(\pi) + log(p_1(x))]}{\partial x}
  \\=
  \frac{log(p_1(x))}{\partial x}
  $$

- $(1 - \pi) p_2(x)$ 的分数：
  $$
  \frac{\partial log((1-\pi) p_2(x))}{\partial x}
  \\=
  \frac{\partial [log(1 - \pi) + log(p_2(x))]}{\partial x}
  \\=
  \frac{log(p_2(x))}{\partial x}
  $$

可以看出，单看其中一个分布的分数时，其分数都和混合比例 $\pi$ 无关。由于Langevin dynamics使用分数进行采样，因此采样出来的样本的分布也会和 $\pi$ 无关。为了验证该分析，如下图所示：

![image-20231218154602915](./imgs/43-Generative Modeling by Estimating Gradients of the Data Distribution/image-20231218154602915.png)

- 图 (a) 是真实的数据分布，采样到左下角的数据的概率应该低于采样到右上角的数据的概率
- 图 (b) 直接用真实的数据分布来计算score，并用朗之万动力学进行采样。可以看出，即使用真实的score，采样出的数据分布也和真实数据分布不一致，两簇数据的概率几乎相同，并不符合真实数据的情况。

# 4 Noise Conditional Score Networks (NCSN)

用随机高斯噪声扰动数据能够使数据更适合基于分数的生成模型：

- 由于高斯噪声是分布在整个编码空间的，因此扰动数据不存在流形假设中所说的真实数据仅存在于低维流形中的问题。
- 高斯噪声还能够填充原始未扰动数据分布中的低密度区域，从而能够进行更准确的分数估计。
- 此外，利用多个水平的高斯噪声进行扰动，能够获得一系列收敛于真实数据分布的噪声扰动的数据分布，利用这些中间分布，可以提高多峰采样（即混合概率分布）的准确率。

基于上述直觉，本文对基于分数的生成模型进行了以下改进：

- 使用不同level的噪声扰乱数据
- 训练一个单个的，有条件（噪声水平）的score估计网络，来估计所有不同噪声水平下的分数。
- 采样时，使用训练时最大的噪声的分数作为初始值，并逐渐退火降低噪声水平。这种方式有助于把大噪声水平的优势平稳的转移到低噪声水平。

## 4.1 NCSN

定义：

- 共有 $1, 2, .., L$ 共 $L$ 个具有不同方差的高斯噪声对原始数据进行扰动。

- 方差 $\{ \sigma_i \}_{i=1}^L$ 表示对应的 $L$ 个不同水平的方差。其中，随着 $L$ 的增大，方差逐渐减小。即， $\sigma_1$ 的方差需要足够大来解决之前描述的低维流形/低密度区域/混合概率分布的问题；且 $\sigma_L$ 的方差需要足够小，使得加了扰动之后的数据与原始数据几乎没有差别。按照论文中的公式，方差逐渐减小的数学描述为：
  $$
  \frac{\sigma_1}{\sigma_2} = ... = \frac{\sigma_{L-1}}{\sigma_L} > 1
  $$

NCSN的目标是训练一个单个的分数估计模型 $s_\theta(x, \sigma)$ ，该模型以噪声水平为条件，通过一个模型实现对任意 $\sigma$ 条件下的分数进行估计：
$$
s_\theta(x, \sigma) \approx \nabla_x logq_\sigma(x)
$$
分数估计模型的输出维度与输入维度相同，即 $x \in \mathbb{R}^D, s_\theta(x, \sigma) \in \mathbb{R}^D$ 。由于输入输出维度相同，NCSN使用U-Net结构并结合空洞卷积；使用IN；通过conditional instance normalization来注入条件 $\sigma_i$ 。

## 4.2 NCSN学习目标

基于前面的描述，由于真实数据分布 $p(x)$ 未知，因此通过加入噪声扰动来把数据转移到预定义的数据分布上来，因此score matching的目标函数为：
$$
J(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}} ||^2]
$$
其次，NCSN加入的是0均值，$\sigma_i$ 方差的高斯噪声，因此扰动后的数据分布为：
$$
q_\sigma(\tilde{x} | x) \in N(\tilde{x}; x, \sigma_i^2I) 
$$

- 高斯分布的概率密度函数表达式是已知的，因此可以计算 $J(\theta)$ 中的对数梯度：

  ![image-20231218161834934](./imgs/43-Generative Modeling by Estimating Gradients of the Data Distribution/image-20231218161834934.png)
  $$
  
  $$

所以，NCSN的学习目标为：
$$
l(\theta; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\tilde{x} - x}{\sigma^2}  ||^2]
$$
$L$ 个噪声水平的统一的损失定义为：
$$
L(\theta; \{ \sigma_i \}_{i=1}^L) = \frac{1}{L} \sum_{i=1}^{L} \lambda(\sigma_i) l(\theta;\sigma_i) 
$$
其中，$\lambda(\sigma_i) > 0$ 是权重函数：

- 实验发现，目标函数中的 $||s(...)||_2 \propto 1/\sigma$ 

- 因此，启发式的设置 $\lambda(\sigma_i) = \sigma^2$ ，从而实现：
  $$
  l(\theta; \sigma) = \frac{1}{2} \sigma^2 \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\tilde{x} - x}{\sigma^2}  ||^2]
  \\=
  l(\theta; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| \sigma s(\tilde{x};\theta) -  \frac{\tilde{x} - x}{\sigma}  ||^2]
  $$

- 这样就同时实现了：

  - $||\sigma s(...)||_2 \propto 1$
  - $\frac{\tilde{x} - x}{\sigma} \in N(0, I)$ 

- 这样做的好处是，对于不同水平的 $\sigma$ ，loss是完全在同一个水平的，因为 loss中的两项与 $\sigma$ 的大小没有任何关系。

最后需要注意，喂给分数网络的样本是每一步的  $\tilde{x}$ 。

## 4.3 NCSN inference via annealed Langevin dynamics

![image-20231218163654239](./imgs/43-Generative Modeling by Estimating Gradients of the Data Distribution/image-20231218163654239.png)

退火Langevin dynamics算法如上图所示：

- 从固定的先验分布开始，初始化 $\tilde{x}_0$ 。
- 从分布 $q_{\sigma_1}(x)$ 开始，获得步长 $\alpha_1$ 
- 对于每个噪声水平，运行 $T$ 个Langevin dynamics采样步，采样出 $q_{\sigma_2}(x)$ 中的分布。
- 之后降低步长到 $\alpha_2$   

由于每一步的数据分布 $\{ q_{\sigma_i} \}_{i=1}^L$ 都是受到一定程度的高斯干扰的数据，因此这些分布能够填满整个空间，并且每个分布的分数都是很好的定义的。

# 5 伪代码

```python
def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    # 取出每个样本对应噪声级别下的噪声分布的标准差，即公式中的sigma_i，
    # 这里的 labels 是用于标识每个样本的噪声级别的，就是 i，实际是一种索引标识
    # (bs,)->(bs,1,1,1) 扩展至与图像一致的维度数
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    # 加噪：x' = x + sigma * z (z ~ N(0,1))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    
    # 目标score，本质是对数条件概率密度 log(p(x'|x)) 对噪声数据 x' 的梯度
    # 由于这里建模为高斯分布，因此可计算出结果最终如下，见前文公式(vii)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    # 模型预测的 score
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    # 先计算每个样本在所有维度下分数估计的误差总和，再对所有样本求平均
    # 见前文公式(vii)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)
```

