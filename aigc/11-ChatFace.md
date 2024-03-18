ChatFace: Chat-Guided Real Face Editing via Diffusion Latent Space Manipulation

项目地址 ：https://dongxuyue.github.io/chatface/

# 1 介绍

尽管扩散模型能够较好的进行图像重建，但是对于基于文字指令的细粒度人脸属性编辑仍然不足。为了解决这个问题，我们提出了一个新的方法，在扩散模型的语义潜在空间中构建文本驱动的图像编辑。通过在扩散模型的生成过程中，使用语义条件对齐扩散模型的时序，我们引入了一种稳定的编辑策略，表现除了精确的零样本编辑效果。

此外，我们开发了一个可交互的ChatFace系统。能够使用户高效的编辑扩散模型的语义隐空间。

----

对于基于文字指令的图像编辑，一个自然的方法是使用CLIP来修改隐编码。然而，我们发现这样做的结果通常不稳定，会带来意想不到的其他变化。为了解决这个限制，我们提出了一种新的人脸编辑pipeline，可以在真实图像中编辑任意人脸属性。

我们从 Diffusion AutoEncoders （DAE）的输入语义编码开始，后见一个映射网络来产生目标编码。随后我们引入了一种 Stable Manipulation Strategy （SMS) 策略，通过将扩散模型的时间特征与语义条件生成过程对齐，在扩散语义空间中进行线性插值，从而实现对真实图像的精确人脸属性编辑。

# 2 方法

![image-20230623153202088](imgs/11-ChatFace/image-20230623153202088.png)

ChatFace的pipeline如上图所示，我们的目标是发开一个多模态的系统，用于真实人脸图像编辑。

## 2.1 Diffusion Probabilistic Model

由于
$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$
因此，
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon }{\sqrt{\bar{\alpha}_t}}
$$
在反向过程中，已知 $x_t$ 和网络预测的 $\epsilon$，可以估计出$x_0$ ：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon(x_t, t) }{\sqrt{\bar{\alpha}_t}}
$$
在DDIM中：
$$
q(x_{t-1}|x_t) \sim N(
\sqrt{\bar{\alpha}_{t-1}} x_0
+
\sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2)
$$
利用估计的 $x_9$ ，就可以采样出 $x_{t-1}$ 。



## 2.2 Semantic Manipulation

![image-20230623162342840](imgs/11-ChatFace/image-20230623162342840.png)

## 2.1 整体结构

整体网络结构如上图所示：

- 对于图像 $x$ ，首先编码到隐空间，定义为 $z \in \mathbb{R}^{512}$ 
- 随后，通过逆向过程，加入contidion $z$ 

由于我们的目标是使得用户能够在真实图像上进行各种任意的属性编辑，因此对于不同真实图像的较大的分布差异，直接将预定于的编辑方向应用于输入图像是非常有挑战性的。因此，我们训练了一个残差映射网络（一个轻量级的MLP）来预测编辑方向 $\Delta z$ ，之后，注入语义编辑offset定义为：
$$
z_{edit} = z + s \times Mapping(z)
$$
其中，$s$ 是一个缩放参数，用于控制编辑强度。训练时，$s$ 设置为1，推理时，$s$ 根据用户的需求来控制编辑的强度。

## 2.2 Mapping Network

Mapping Network的结构非常简单且轻量，是一个4层MLP，结构如下图所示：

![image-20230624003143785](imgs/11-ChatFace/image-20230624003143785.png)

## 2.2 Stable Manipulation Strategy

在扩散模型初始的去噪阶段，通常捕捉高级特征，如结构和形状；在之后的step中，生成低级特征，如颜色和纹理。因此，对于隐编码$z_t$ ，在不同的去噪阶段都是用相同的语义条件$z_{edit}$ 时不合理的，可能导致损失原始图像的高频细节，最终导致生成结果不稳定。

为了解决上述问题，本文提出了一种插值策略，用于对齐扩散模型中的时序特征，如上图所示：

- 从原始图像获得 $z$ 

- 之后使用之前提到的残差映射网络计算$z_{edit}$ 

- 最后，使用 lerp 线性插值的方法，对 $z_{edit}$ 和 $z$ 进行插值，比例为 $v$ ，得到 $z_t$ ：
  $$
  z_t = Lerp(z_{edit}, z; v)
  \\=
  (1-v) z_{edit} + v z
  $$
  其中：

  ![image-20230623224939057](imgs/11-ChatFace/image-20230623224939057.png)

$v = t/T$ ，$t \in [0, 1, 2, ..., T]$ ，$T$ 时生成的time steps。

可以看出：

- 在去噪早期，$1- v$ 较大，$z$ 主要受残差映射后的 $z_{edit}$ 影响较大
- 在去噪后期，$z$ 主要受原始图像编码的 $z$ 的影响较大，用于恢复颜色纹理等细节信息

$z_t$ 作为扩散模型的条件，因此扩散模型的输出 $\epsilon$ 此时为 $\epsilon(x_t, t, z_t)$  。当设置 DDIM 的 $\eta = 0$ , 则方差为0，即论文中的：

![image-20230623230617476](imgs/11-ChatFace/image-20230623230617476.png)

其中，$f_\theta(x_t, t, z_t)$ 为估计的 $x_0$ 。

## 2.3 Training Objectives

为了获得细粒度的人脸属性编辑，我们提出了三种类型的损失进行约束：

### 2.3.1 重建损失

为了保证扩散模型生成的编辑后的图像和原始图像接近，需要计算重建损失：
$$
L_{pre} = ||x_0 - D(z_{edit})||_1 + ||\Delta z||_2
$$
其中，$D(z_{edit})$ 是扩散模型DDIM，在condition为$z_{edit}$的输出，需要与原图计算重构损失。

$\Delta z = z_{edit} - z_0$ ，用于约束残差映射网络的输出隐编码与原始隐编码的差异不能太大，也是为了保证最终输出的图像与输入图像的一致性。

### 2.3.2 人脸一致性损失

由于我们的任务是编辑人脸属性，同时保证人脸身份的一致性，因此还计算了一个人脸识别损失：
$$
L_{id} = 1 - cos\{ R(D(z_0), R(D(z_{edit})))\}
$$
其中，$R(\cdot)$ 表示预训练的ArcFace网络。

### 2.3.3 CLIP损失

由于使用基于文字指令的方式来编辑图像，因此需要计算CLIP损失来匹配文本和图像特征:
$$
L_{direction} (D(z_{edit}),y_{tar}; D(z_0), y_{ref}) = 1 - \frac{<\Delta I ,\Delta T>}{||\Delta I|| ||\Delta T||}
$$
其中，

- $<,>$ 表示内积。
- $y_{tar}, y_{ref}$ 分别表示两种不同面部属性所对应的文本
- $D(\cdot)$ 表示扩散模型使用不同隐编码 $z$ 的输出
- $\Delta I = E_I (D(z_{edit}) ) - E_I (D(z_0))$ ，其中 $E_I (\cdot)$ 表示 CLIP的图像编码器，$\Delta I$ 表示两种人脸属性的向量
- $\Delta T = E_T (y_{tar}) - E_T (y_{ref})$ ，其中 $E_T (\cdot)$ 表示 CLIP的文本编码器，$\Delta T$ 表示两种人脸属性所对应的文本的向量

因此，CLIP损失项的目的为：使得编辑前后的图像向量和文本向量的相似度尽可能大。



### 2.3.4 Total Loss

最终的loss为：
$$
L_{total} = \lambda_{pre} L_{pre} + \lambda_{id} L_{id} + \lambda_{dir} L_{direction}
$$
实验中，$\lambda_{pre} = 0.2, \lambda_{id} = 0.5, \lambda_{dir}=2.0$

# 3 实验

## 3.1 与其他方法对比

![image-20230624002005531](imgs/11-ChatFace/image-20230624002005531.png)

# 4 实现细节

- 使用预训练的 Diffusion Autoencoders (DAE) ，分辨率 256，用于图像编码和生成。
- 语义编码 $z \in \mathbb{R}^{512}$ ，初始噪声的尺度与输入尺度相同，为 $x_T \in \mathbb{R}^{256 \times 256 \times 3}$ 。
- 为了证明ChatFace的鲁棒性和生成能力，mappig network 使用 CelebA-HQ 数据集上训练。DAE 在 FFHQ 数据集上训练
- 对于人脸图像，实验中使用了 54 中文本prompts，包含了表情，发型，年龄，性别，风格，眼镜等等。
- 使用了 Ranger 优化器，学习率为 0.2，每种属性训练 10000 个 iters，batch_size 为8
- 使用8卡 3090 GPU训练
- 生成图像时，设置 $T = 8$ 
- 对于大语言模型，使用 $GPT-3.5-turbo$ 模型，可以从 OpenAI 的API中获取

## 4.1 用户输入

控制人脸属性编辑的变量有3个：

- 想要编辑的属性 $A$ ，用于文本
- 编辑的强度 $S$ ，用于调节 Mapping 网络的输出权重
- 扩散模型的采样步长 $T$ 

为了从用户输入的一段话中获取上述三个属性，我们给LLM注入的编辑属性，由大模型输出A,S,T。

![image-20230624003327279](imgs/11-ChatFace/image-20230624003327279.png)

调教的过程如下：

![image-20230624004224080](imgs/11-ChatFace/image-20230624004224080.png)

## 4.2 扩散模型生成步长

如下图所示，较少的 $T$ 生成的图像的编辑强度更强，但是缺少了原始图像的高频信息。当$T$ 更大时，采样时间较长。因此使用 $T=8$ 用于平衡。

![image-20230624003747810](imgs/11-ChatFace/image-20230624003747810.png)

![image-20230624003941727](imgs/11-ChatFace/image-20230624003941727.png)

## 4.3 不同强度的 S

![image-20230624004315398](imgs/11-ChatFace/image-20230624004315398.png)

## 4.4 多个属性的组合

![image-20230624004343228](imgs/11-ChatFace/image-20230624004343228.png)
