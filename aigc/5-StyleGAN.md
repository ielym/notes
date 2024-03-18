A Style-Based Generator Architecture for Generative Adversarial Networks

# 1 

受风格迁移的启发，本文重新设计了生成器的结构来控制图像生成的过程：生成器从一个固定的输入开始，基于隐编码在每个卷积层调整输入的风格。因此可以是先在不同特征尺度上来直接控制特征的强度。

没有修改判别器和损失函数。

# 2 Style-based Generator

![image-20230530224105545](imgs/5-StyleGAN/image-20230530224105545.png)

传统的方法：如上图a所示，从一个隐编码 $z$ 开始，通过输入层进行维度变换并提供给生成器。

本文的方法：省略掉输入层，直接从一个可学习的空间分辨率是 $4 \times 4$ 的常量开始，如上图b的右侧所示。

---

- 对于一个隐编码 $z$ ，其隐空间表示为 $Z$ ，一个非线性映射的网络 $f : Z \to W$ 实现 $Z$ 空间到 $W$ 空间的映射，并产生 $w \in W$ 。为了简化起见，我们设置 $z， w$ 的维度都是 512-d 。映射函数 $f$ 是一个 8 层的感知机。

- 之后，一个可学习的仿射变换把 $w$ 变换成风格 styles $y = (y_s, y_b)$ ，以此来控制 Adaptive Instance Normalization (AdaIN) 。AdaIN定义为：
  $$
  AdaIN(x_i, y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b, i}
  $$
   其中，每一个 $x_i$ 单独进行归一化，并使用 style $y$ 来缩放和平移。因此，$y$ 的维度是该层特征图的维度的两倍。

- 最后，通过引入显式的噪声输入，给生成器引入随机性。这些噪声输入是单通道的图像，不相关的高斯噪声。噪声图像被传播给所有的特征图，使用一个可学习的 per-feature的缩放因子，之后与对应卷积层的输出相加。

## 2.1 Quality of generated images

![image-20230815190553188](imgs/5-StyleGAN/image-20230815190553188.png)

如上图所示：

- Baseline是 Progressive GAN
- B ：使用双线性插值进行上采样/下采样，更长的训练步数，微调超参数
- C ：加入Mapping Network和AdaIN算子
- D ：移除传统方法的输入层，直接从一个可学习的 $4 \times 4 \times 512$ 的常量tensor开始
- E ：加入噪声输入
- F ：使用 mixing regularization

# 3 Properties of the style-based generator

从styleGAN的结构可以看出，mapping network和仿射变换用于从一个学习到的分布中采样各种风格，此外，由于不同style作用于不同的网络层，因此这种结构能够准确的控制图像生成。如，修改一个特定的style，可以对应于修改图像上特定的地方。这种能力的解释如下：

- AdaIn首先把特征的每个通道归一化成0均值，单位方差。之后，使用styles的scale和bias作用于特征上，重新进行缩放和平移。
- AdaIN输出的特征具有新的逐通道的统计特性，scale和bias也能够修改不同通道的特征图的重要程度。

## 3.1 Style mixing















