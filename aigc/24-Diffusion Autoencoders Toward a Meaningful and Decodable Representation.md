`Diffusion Autoencoders: Toward a Meaningful and Decodable Representation`

code : https://Diff-AE.github.io/



# Motivation

DPMs 和 GAN 不同，隐变量缺乏语义信息。

本文研究是否能够通过autoencoder对图像提取有意义并且能够解码的能力。主要想法是使用一个科学系的编码器来提取高级语义，并使用DPMs作为解码器来解码。

该方法能够把任意图像编码成两部分隐编码：

- 一部分是语义
- 另一部分捕捉随机细节，用于近乎精确的

这些能力使得DPMs具有当前基于GAN的方法的能力，如属性编辑。此外我们也证明了这种两级编码改善了去噪效率。

为了找到对图像有意义的表示，需要包含高级语义和低级随机变量。本文的方法是使用一个可学习的编码器来提取提取高级语义信息，使用一个DPM来提取并建模随机变量：

- 使用CNN作为编码器，来提取高级语义特征
- 使用DDIM的逆过程来获取随机变量



在DDIM的采样过程中，推导过：
$$
q(x_{t-1}|x_t) \sim N(
\sqrt{\bar{\alpha}_{t-1}} x_0
+
\sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2)
$$
此外，对于一张输入图像 $x_0$ ，可以把 DDIM 当成一个编码器，非常准确的获得 $x_T$ ，然而，$x_T$ 仍然不包含高级语义信息。为了证明这一点，我们做了如下实验：

- 对于两张图像，分别采样出各自的 $x_T$ 
- 对两个图像的 $x_T$ 进行插值，然后执行反向过程，结果如下：

![image-20230805200439284](imgs/24-Diffusion%20Autoencoders%20Toward%20a%20Meaningful%20and%20Decodable%20Representation/image-20230805200439284.png)

从上图可以看出，最左边和最右边的两个原始图像插值之后，中间的图像只共享了整体构图结构和背景颜色，但并没有融合两个人的身份。

# Diffusion autoencoders

我们设计了一个条件DDIM图像解码器 $p(x_{t-1}|x_t, z_{sem})$ ，

其中，$z_{sem}$ 是表示语义的隐变量，用于控制解码过程。 
$$
z_{zem = Enc_\Phi (x_0)}
$$


$Enc_\Phi (\cdot)$ 是一个语义编码器，用于学习从原始输入图像 $x_0$ 到语义编码 $z_{sem}$ 的映射。

条件 DDIM 解码器使用隐变量 $z = (z_{sem}, x_T)$ 作为输入，包含了高级语义信息 $z_{sem}$ 和低级随机信息 $x_T$ 。$x_T$ 通过DDIM的逆过程获得。

整体结构如下图所示：

![image-20230805182620948](imgs/24-Diffusion%20Autoencoders%20Toward%20a%20Meaningful%20and%20Decodable%20Representation/image-20230805182620948.png)

和其他扩散概率模型使用空间条件变量（如2D的latent maps），我们的 $z_{sem}$ 是一个非空间的向量，维度 $d = 512$， 这点和 StyleGAN 一致，允许我们可以把任意空间分辨率的图像编码成全局语义。

我们的目标是学习一个语义丰富的隐空间，并可以进行平滑插值，类似于GAN。同时保持重建能力。

## Semantic Encoder

语义编码 $Enc(x_0)$ 的目的是把输入图像转换成包含语义信息的向量 $z_{sem} = Enc(x_0)$ ，用于之后Diffusion-based Decoder 的去噪过程 $p_\theta(x_{t-1} | x_t, z_{sem})$ 。本文 $Enc(\cdot)$ 的结构和UNet的编码器的结构相同，没有专门的设计。有了语义编码 $z_{sem}$ 的加持，DDIM的去噪过程会更高效（之后会实验介绍）

## Stochastic Encoder

Stochastic的目的是给一张原图，计算出该图像的 $x_T$ ，用于之后 DDIM Decoder 去噪。与其他方法不同的是， $x_T$ 并不是随机加噪，而是不断的根据 $x_0$ 计算出 $x_{t+1}$ ，直至 $x_T$ 。

本文反向加噪使用的是 DDIM Inverse （Diffusers里有该方法，但是稍微有所不同，后面会说）。

具体方法如下：

根据
$$
q(x_{t-1}|x_t) \sim N(
\sqrt{\bar{\alpha}_{t-1}} x_0
+
\sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2)
$$
可知：
$$
x_{t-1} =

\sqrt{\bar{\alpha}_{t-1}} x_0
+
\sqrt{1 - \bar{\alpha}_{t-1}} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}
$$
把 $x_t$ 变换到等式左边：
$$
\sqrt{1 - \bar{\alpha}_{t-1}} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}

=

\sqrt{\bar{\alpha}_{t-1}} x_0
-
x_{t-1}
$$

$$
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}} 

=

\frac{\sqrt{\bar{\alpha}_{t-1}} x_0
-
x_{t-1}}{\sqrt{1 - \bar{\alpha}_{t-1}}}
$$

$$
x_t - \sqrt{\bar{\alpha}_{t}} x_0

=
\sqrt{1 - \bar{\alpha}_t}

\frac{\sqrt{\bar{\alpha}_{t-1}} x_0
-
x_{t-1}}{\sqrt{1 - \bar{\alpha}_{t-1}}}
$$

$$
x_t 

=
\sqrt{1 - \bar{\alpha}_t}

\frac{\sqrt{\bar{\alpha}_{t-1}} x_0
-
x_{t-1}}{\sqrt{1 - \bar{\alpha}_{t-1}}}

+
\sqrt{\bar{\alpha}_{t}} x_0
$$

由 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon(x_t, t)$ 可知：
$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon(x_{t-1}, t-1)
$$

$$
x_0 = 
\frac
{x_{t-1} - \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon(x_{t-1}, t-1)}

{\sqrt{\bar{\alpha}_{t-1}}}
$$



带入：
$$
x_t 

=
\sqrt{1 - \bar{\alpha}_t}

\frac{\sqrt{\bar{\alpha}_{t-1}} x_0
-
x_{t-1}}{\sqrt{1 - \bar{\alpha}_{t-1}}}

+
\sqrt{\bar{\alpha}_{t}} x_0

\\=

\sqrt{1 - \bar{\alpha}_t}
\frac{x_{t-1} - \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon(x_{t-1}, t-1) - x_{t-1}}{\sqrt{1 - \bar{\alpha}_{t-1}}}

+
\sqrt{\bar{\alpha}_{t}} x_0
$$






## Diffusion-based Decoder

和DDIM类似，解码器的损失计算为：
$$
L_{simple} = \sum_{t=1}^{T} \mathbb{E}_{x_0, \epsilon_t} 

[

|| \epsilon_\theta(x_t, t, z_{sem}) - \epsilon_t ||_2^2

]
$$

- 为了把条件 $z_{sem}$ 加到UNet的去噪过程中，使用一个仿射变换把 $z_{sem}$ 变换成UNet的特征通道数 : 

$$
z_s = Affine(z_{sem}) \in \mathbb{R}^c
$$

- 为了把时间步 $t$ 加到UNet的去噪过程中，使用MLP进行变换：
  $$
  (t_s, t_b) = MLP(\Psi(t)) \in \mathbb{R}^{2 \times c}
  $$
  其中，$\Psi(t)$ 是正弦变换。

- 对于UNet的特征 $h$ ，$z_{sem}$ 和 $t$ 使用 AdaGN 对 $h$ 进行放射变换来实现，与 StyleGAN 类似：
  $$
  AdaGN(h, t, z_{sem}) = z_s (t_s GroupNorm(h) + t_b)
  $$

# Sampling with Diffusion AutoEncoders

通过条件