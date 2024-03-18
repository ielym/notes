Adding Conditional Control to Text-to-Image Diffusion Models

# 1 摘要

本文提出了一个神经网络结构 ControlNet，用于控制训练好的大型扩散模型，来支持额外的condition输入。ControlNet通过端到端的学习task-specific的conditions，即使训练数据较少（<50k），训练依然是鲁棒的。此外，训练ControlNet就像fine-tune一个扩散模型一样快，可以在个人的设备上训练模型。此外，如果计算资源足够，模型也可以缩放到大量数据。

# 2 介绍

作者研究了不同的图像处理应用，有三点发现：

- 特殊任务领域的数据不像通用image-text领域的数据规模这么大，特殊领域的数据量通常少于 100k，比如，比LAION-5B少了 $5 \times 10^4$ 倍。这需要鲁棒的神经网络训练方法来防止过拟合，并保持生成能力。
- 当图像处理任务是数据驱动的解决方案时，不是都有大规模的计算集群。这使得在能够接收的时间和内存空间下，更快的优化大模型是非常重要的。
- 不同图像处理有不同形式的问题定义。如depth-2-image, pose-2-human等任务，这些问题本质上需要将原始输入解释为对象级或场景级理解，这使得手工制作的程序方法不太可行。为了在许多任务中获得学习解决方案，端到端学习是必不可少的。

因此本文提出了ControlNet，通过控制大图像扩散模型，来学习task-specific的输入conditions。ControlNet拷贝两份之前训练好的模型权重，分别作为 trainable copy 和 locked copy：

- locked copy ：不更新权重
- trainable copy ：学习task-specific的数据，更新权重。

# 3 方法

## 3.1 ControlNet

![image-20230612233537203](imgs/8-ControlNet/image-20230612233537203.png)

- 对于一个训练好的block，分别拷贝一份 locked 和 一份 trainable 
- trainable block的参数使用条件向量 $c$ 来训练
- 之所以训练 trainable 而不是直接更新block，是为了防止在小数据集上过拟合，同时保留大模型在billions of images上的训练的质量。

block通过 "zero convolution" ，如 1x1的卷积层来连接。zero convolution 的weight和bias初始化为0。用 $Z(\cdot;\cdot)$ 来表示zero convolution，两个zero convolution的权重分别为 $\Theta_{z1}, \Theta_{z2}$ ，$F(x; \Theta)$ 为locked的blocked的输出，$F(x; \Theta_c)$ 是trainable block的输出，则上图右侧整个block的输出为：
$$
y_c = 
F(x; \Theta)
+
Z( F(x + Z(c; \Theta_{z1}); \Theta_c); \Theta_{z2} )
$$
 由于 $\Theta_{z1}, \Theta_{z2}$ 都初始为0，因此第一次训练的前向传播时，$y_c = y$ 。

## 3.2 ControlNet in Image Diffusion Model

![image-20230612235018012](imgs/8-ControlNet/image-20230612235018012.png)

本文使用Stable Diffusion作为示例，来介绍使用ControlNet来控制扩散模型的方法：

- SD使用类似VG-GAN的方法来把 $512 \times 512$ 的图下采样成 $64 \times 64$ 
- 因此 ControlNet需要把 image-based condition 也转换成 $64 \times 64$ 的特征空间，以匹配SD的特征。本文使用一个tiny的网络 $E(\cdot)$ 来实现，包含4个卷积层，每个卷积的步长是 $2 \times 2$ ，并使用ReLU作为激活函数，通道数分别为16, 32, 64, 128，使用高斯分布初始化权重，$E(\cdot)$ 与整个网络共同优化，来得到编码 $c_f = E(c_i)$ 

## 3.3 Improved Training

### 3.3.1 Small-Scale Training

上图所示的网络结构。连接了ControlNet 和 SD Middle Block ， SD Decoder Block 1,2,3,4 。

实验发现，断开 decoder 1,2,3,4的连接，只保留 middle block 的连接，可以在 RTX 3070TI 的笔记本电脑上的GPU上加速1.6倍。

当模型训练到一定程度后，显示出结果和条件之间的合理关联时，这些断开的环节可以在继续训练中重新连接起来，以方便精确控制。

### 3.3.2 Large-Scale Training

如果有强大的计算集群（至少8卡A100 80G或等价的资源），以及大数据（至少 1M）可用，过拟合的风险很低，因此可以使用 Large-Scale Training。

- 首先，训练ControlNet足够多的iterations（通常大于50k steps）
- 之后，解冻Stable Diffusion的所有权重，并联合训练整个模型。