High-Resolution Image Synthesis with Latent Diffusion Models

# 1 动机

扩散模型在生成结果上取得了sota结果，此外，扩散模型允许使用过一个引导机制来控制图像生成的过程，而不需要重新训练。

然而，扩散模型通常直接在像素空间操作，优化一个强大的扩散模型需要花费数百个GPU-days，并且由于序列化的推理过程，推理也很慢。

为了使能够在有限的计算资源下训练扩散模型，同时保持其质量和灵活性，我们在隐空间（latent space）来使用扩散模型。

通过引入cross-attention层到模型结构中，能够使扩散模型对于通用的条件输入下变得强大和灵活，如，可以输入文本，边界框，高分辨率的生成变得可能。

本文提出的Latent-Diffusion模型（LDMs）在多个任务上取得了新的sota效果。

# 2 介绍

扩散模型属于likelihood-based模型，速度较慢。尽管一些方法通过 reweighted variational objective来试图通过下采样输出的去噪steps来加速，但是仍然需要大量计算资源。这是由于训练和推理都是在高纬度的RGB图像空间进行的。

本文把训练过程分成两个部分：

- 首先，训练一个 autoencoder，用于提供一个低维（因此高效）的表征空间。
- 之后，在学习到的latent space中训练扩散模型。

该方法有一个显著的优点：我们只需要训练一次autoencoder，之后通过训练多个DM模型用于不同任务。这使得可以高效的探索不同的Diffusion模型，用于不同的图生图和文生图任务中。

# 3 方法

## 3.1 Perceptual Image Compression

对于$x \in \mathbb{R}^{H\times W \times 3}$ 的RGB图像，编码器 $E$ 把 $x$ 编码到一个latent representation $z = E(x)$ ，解码器 $D$ 从latent representaion 重建图像 $\tilde x = D(z) = D(E(z))$ 。其中，$z \in \mathbb{R}^{h \times w \times c}$ ：

- 编码器通过下采样因子 $f = \ H/h = W/w$ 来下采样图像，本文后面讨论了不同的 $f = 2^m$ 
- 为了比避免高方差的隐空间，文中实验了两种不同的正则化 KL-reg 和 VQ-reg。其中KL-reg能够使隐空间变成一个标准正态分布的空间，类似于VAE。VQ-reg来源于VQGAN。

## 3.2 Latent Diffusion Models

### 3.2.1 Diffusion Models

扩散模型是一个概率模型，通过逐步的降噪一个正太分布的噪声来学习数据分布 $p(x)$ 。对应的目标函数可以简化为：
$$
L_{DM} = \mathbb{E}_{x,\epsilon \sim N(0, 1), t} [||\epsilon - \epsilon_\theta(x_t, t)||_2^2]
$$

### 3.2.2 Generative Modeling of Latent Representations

使用训练好的感知压缩模型 $E$ 和 $D$ ，我们现在有一个高效的，低维的隐空间。隐空间去除了高频和细节信息。对比高维的像素空间，隐空间有两个优点：

- 更关注重要的和强语义信息
- 在低维度训练更加高效

对应的目标函数为：
$$
L_{DM} = \mathbb{E}_{E(x),\epsilon \sim N(0, 1), t} [||\epsilon - \epsilon_\theta(z_t, t)||_2^2]
$$
其中，$\epsilon_\theta(\cdot, t)$ 是一个time-conditional 的 UNet 。训练时 $z_t$ 可以从 $E$ 中获取。

## 3.3 Conditioning Mechanisms

类似于其他生成模型，扩散模型通过建立条件分布$p(z|y)$ 来控制生成。这种方法可以通过一个条件去噪自编码器 $\epsilon_\theta(z_t, t, y)$ 来实现，其中 $y$ 可以时文本，语义图，或其他image-to-image的任务。

我们使用cross-attention机制，通过增强UNet backbone来实现更加灵活的条件控制：

- 为了预处理不同模态的 $y$ ，我们引入了一个 domain specific 的编码器 $\tau_\theta$ ，用于把 $y$ 转换成中间形式的表达 $\tau_\theta (y) \in \mathbb{R}^{M \times d_\tau}$ 。
- 之后，通过 cross-attention 把 $\tau_\theta (y)$ 映射到Unet的中间层 ：
  - $Q = W_Q^{(i)} \cdot \phi_i(z_t)$ ，$K = W_K^{(i)} \cdot \tau_\theta(y)$ ，$V = W_V^{(i)} \cdot \tau_\theta(y)$
  - $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}} \cdot V)$
  - 其中，$\phi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon^i}$ 表示UNet flattened 的中间特征，$W_v^{(i) \in \mathbb{R}^{d \times d_\epsilon^{i}}}$  ，$W_Q^{(i) \in \mathbb{R}^{d \times d_\tau}}$ ，$W_K^{(i) \in \mathbb{R}^{d \times d_\tau}}$ 是投影矩阵。

目标函数为：
$$
L_{LDM} := \mathbb{E}_{E(x), y,\epsilon \sim N(0, 1), t} [||\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))||_2^2]
$$
$\tau_\theta, \epsilon_\theta$ 通过上式联合优化

# 4 实验

## 4.1 On Perceptual Compression Tradeoffs

该节对比不同下采样因子 $f \in \{1, 2, 4, 8, 16, 32\}$ 的效果和速度。

![image-20230612093322409](imgs/7-StableDiffusion/image-20230612093322409.png)

- 如上图所示，为了达到相同的 FID或Inception Score，LDM-1需要更长的tran step。
- 过大的下采样倍数，如LDM-32在训练一定step之后，效果会停滞不再改进。作者认为是在第一个阶段国强的压缩导致信息损失，从而影响了质量。
