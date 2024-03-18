`IP-Adapter Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models`

# 1 动机

- 文本条件的表达能力不足，需要复杂的prompt工程。而图像能够表示的信息远多于文本，“an image is worth a thousand words”
- 现在虽然已经有把图像编码嵌入到扩散模型中的工作，但通常是简单的拼接文本和图像特征（图像特征通常用CLIP获取），但是这些方法的效果通常不足。
  - 作者认为这个问题是由于：之前的方法依赖于文生图扩散模型的cross-attention，虽然CLIP的文本编码器和图像编码器的输出特征相似度足够高，但是 K, V 是基于文本特征进行训练的，强行使用拼接的方法加入图像特征会潜在的丢失一些图像特有的信息，因此只能粗粒度的控制生成（如，图像风格）

本文提出的 IP-Adapter 只有22M参数。



# 2 方法

![image-20230911225551277](imgs/30-IP-Adapter%20Text%20Compatible%20Image%20Prompt%20Adapter%20for%20Text-to-Image%20Diffusion%20Models/image-20230911225551277.png)

## 2.1 先验知识

有条件扩散模型的损失为：
$$
L_{simple} = \mathbb{E}_{x_0, \epsilon \in N(0, I), c,t} || \epsilon - \epsilon_\theta(x_t, c, t) ||^2
$$
其中，$c$ 是额外的条件

classifier-free中，$\hat{\epsilon}_\theta$ 为：
$$
\hat{\epsilon}_\theta(x_t, c, t) = w \epsilon_\theta(x_t, c, t) + (1 - w)\epsilon_\theta(x_t, t)
$$
w是guidance scale

## 2.2 IP-Adapter

### 2.2.1 Image Encoder

- 使用的图像编码器是  OpenCLIP ViT-H/14

- 如上图所示，使用一个线性层和LN层把CLIP的输出特征映射成 $N=4$ 的序列，每个序列的维度和文本编码维度相同（768）

### 2.2.2 Decoupled Cross-Attention

如动机分析，图像和文本的语义不同，因此使用了解耦的Cross-Attention。

- 文本的cross-attention不变：
  $$
  Z' = Attention(Q, K, V) = softmax( \frac{QK^T}{\sqrt{d}} )V
  $$
  其中，$Q = ZW_q$ ，$K = c_tW_k$ ，$V = c_tW_v$ 。$c_t$ 是文本编码。
  
- 图像cross-attention与文本注意力类似：
  $$
  Z^{''} = Attention(Q, K',V') = softmax( \frac{Q(K^{'})^{T}}{\sqrt{d}} )V'
  $$
  其中，$Q = ZW_q$ ，$K = c_i W'_k$ ，$V = c_i W'_v$ 。$c_i$ 是图像编码。

- NOTE: 图像编码cross-attention的 $Q$ 与文本编码的相同，因此，只有 $W'_k, W'_v$ 是额外引入的参数。

- 最后，两个cross-attention的输出直接相加：
  $$
  Z^{new} = softmax( \frac{QK^T}{\sqrt{d}} )V + softmax( \frac{Q(K^{'})^{T}}{\sqrt{d}} )V'
  $$

### 2.2.3 训练&&推理

#### 2.2.3.1 损失函数

需要训练和冻结的参数结构如图所示。损失函数为：
$$
L_{simple} = \mathbb{E}_{x_0, \epsilon \in N(0, I), c_t, c_i,t} || \epsilon - \epsilon_\theta(x_t, c_t, c_i, t) ||^2
$$
无条件时：
$$
\hat{\epsilon}_\theta(x_t, c_i, c_t, t) = w \epsilon_\theta(x_t, c_i, c_t, t) + (1 - w)\epsilon_\theta(x_t, t)
$$
其中，无条件时文本编码和图像编码都会随机置空：

- 当图像编码置空时，CLIP的输出全部置0

#### 2.2.3.2 推理

由于图像和文本的cross-attention是独立的，因此推理时可以使用权重控制两个注意力的融合比例：
$$
Z^{new} = Attention(Q, K, V) + \lambda Attention(Q, K',V')
$$
当只用文本编码时，$\lambda = 0$ 

# 3 实验

## 3.1 训练数据

- 来自 LAION-2B 和 COYO-700M 的10M个图像文本对

## 3.2 实现细节

- 模型：SD1.5 （文本编码器+.....） + OpenCLIP ViT-H/14 （图像编码器） 

- SD1.5的每个cross-attention都加了一个图像注意力，参数仅有22M
- 8*V00,1M steps， bs=8，AdamW，固定学习率 1e-4，weight decay = 1e-2
- 最短边resize 512，中心裁剪512x512
- 0.05的概率独自的舍去text和图像。0.05的概率同时舍去文本和图像。
- 推理时，DDIM 50步，guidance scale = 7.5
- 只使用文本编码时，$\lambda = 0$ ，只使用图像编码时，文本为空且$\lambda = 1.0$ 

