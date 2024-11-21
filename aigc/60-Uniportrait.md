# 0 引用

- [UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization](https://arxiv.org/abs/2408.05939)

# 1 介绍

![image-20241120231511265](imgs/60-Uniportrait/image-20241120231511265.png)

- 支持单图或多图输入
- 目前training-free的方法存在两个缺点：
  - 一些使用global embedding的方法会损失空间信息，同时一些使用CLIP的方法不适合处理人脸特征
  - 基于encoder的方法难以控制和编辑人脸。
  - 论文认为这是由于目前的方法无法解耦人脸一致性和人脸相关但身份无关的表征导致的，因此会过拟合到参考人脸。
- 此外，现有方法主要关注单人脸，无法处理多人脸场景：主要存在身份混淆的问题
  - 一些方法把ID信息聚合到文本编码中，并使用文本来区分不同主题。但这类方法需要一对一的文本映射，缺乏free-form text prompts的任务，如 two women。此外，这种方法由于图像和文本自身的差异，也会影响语义或一致性。
  - 另一类方法尝试使用Mask来区分不同id，但需要指定面部区域。
- 为解决上述问题，提出了UniPortrait：
  - 联合单人脸和多人脸，同时具有面部编辑能力，free-form的输入形式，多样的layout生成（不受mask区域约束）
  - 包含一个 ID embedding module 和一个 ID routing module：
    - ID embedding module 对每个人脸提取面部特征，并编码到文本空间
    - ID routing module 自适应的把embeddings注入到各自的区域中。
  - 没有使用last global feature，而是使用人脸识别backbone的倒数第二层的具有空间结构的特征
  - 为了增强人脸一致性，同时使用了CLIP的shallow local features
  - 为了解决过拟合光照，姿态等信息，通过强dropping的正则方式，显式的解耦人脸结构特征，如 DropToken & DropPath
  - ID routing module对于cross-attn中的每个空间位置都预测一个离散概率分布，并挑选最匹配的top-1 ID embedding来注入到特定的位置：
    - 为了保证所有的ID都会被router到，并且保证每个ID只会被router到一个区域。引入了routing regularization loss

# 2 方法

## 2.1 ID Embedding Module

![image-20241121000321108](imgs/60-Uniportrait/image-20241121000321108.png)

- 如上图所示，使用人脸识别的倒数第二层的特征（1/16），使用一个MLP来获取ID特征 $F_r \in \mathbb{R}^{m_r \times d_r}$ ，$m_r$ 表示特征长度，$d_r$ 表示特征维度。
- 同时，使用人脸识别的 $1/2, 1/4, 1/8$ 的特征，进行插值。之后和CLIP的特征concat起来，过另外一个MLP，得到 $F_s \in \mathbb{R}^{m_s \times d_s}$ 。
- 之后，使用一个 $l$ 层的Q-Former来query $F_r, F_s$ 。每层包含两个attention blocks，以及一个FFN，query的token数为 $m$ 。QFormer的输出为 $F_{id} \in \mathbb{R}^{m \times d}$ ，作为最终的ID Embedding，并对齐U-Net的文本空间。
  - 在第二个attention block的输入和输出的地方，分别使用了DropToken和DropPath。
- 使用IP-adapter的cross-attention的方式注入qformer的特征。

### Single-ID Multi-Reference Embedding

![image-20241121001952828](imgs/60-Uniportrait/image-20241121001952828.png)

- 对于同一张人脸的多张参考图，每张图都提取 $F_{r(j)}, F_{s(j)}$ ，$j$ 为参考图索引。
- 之后直接全部都concat起来使用
- 实验发现，尽管在训练时只使用单张参考图，但是推理时可以有效扩展到多张参考图来改进个性化生图结果。

### Identity Interpolation

- 可以对多个ID进行插值。
- 对于每个id的qformer结果 $F_{id}^{(n)}$ ，$n$ 表示id索引，使用线性插值

## 2.2 ID Routing

![image-20241121003133382](imgs/60-Uniportrait/image-20241121003133382.png)

- 每个ID Embedding Module可以获取单个ID的 ID embedding。

- 对于 $N$ 个不同的IDs，每个ID的 ID embedding 为 $F_{id}^{(n)}, n=1,2,...,N$ 。

- 对于生成图像 $Z \in \mathbb{R}^{c \times h \times w}$ 上的每个空间位置 $(u, v)$  ，我们指定一个单独的ID $k^*$ 进行注入，$ 1 \le k^* \lt N$ ：
  $$
  k^* = \underset{k} {\arg \max}\,  \psi (Z, F_{id}^{(:)}, (u, v))_k
  $$

  - $\psi$ 表示router，输出一个 N-dim的离散概率分布：
    $$
    \psi (Z, F_{id}^{(:)}, (u, v)) = Softmax([\theta(Z)_{u,v} * \phi(W_{aggr} * F_{id}^{(n)}) ]_{n=1}^{N})
    $$

    - $W_{aggr} \in \mathbb{R}^m$ ，把每个id的 $F_{id}^{(n)} \in \mathbb{R}^{m \times d}$ 的token维度聚合成一个单独的token。
    - $\theta, \phi$ 分别是一个2 层的MLP，$*$ 是矩阵乘法操作。 

- 通过ID Routing，保证每个像素只能query到一个id，避免了多id属性混淆的问题。然而，上面的方法存在三个问题：

  - 没有办法保证同一个像素只匹配到一个id：
    - 使用routing regularization loss
  - 没有办法保证所有id都能被匹配到
    - 使用routing regularization loss
  - argmax不可微
    - 使用Gumbel softmax trick

### Routing Regularization Loss

- 对于target图像，首先检测出所有人脸，并转化成binary masks。1表示人脸区域，0表示非人脸区域。
  - 每个人脸一个单独的mask，N 个人脸一共有 N 个 binary masks。

$$
L_{route} = \lambda \frac{1}{N} || W_{route} \odot (\psi (Z, F_{id}^{(:)} ) - M)  ||_2^2
$$

- $\lambda$ 表示loss权重，$\odot$ 表示element-wise multiplication
- $\psi (Z, F_{id}^{(:)}) \in \mathbb{R}^{N \times h \times w}$ ，表示所有位置，所有N个id的softmax值 
- $M \in \mathbb{R} ^ {N \times h \times w}$ 表示N个id的binary masks gt
- $W_{route} \in \mathbb{R}^{h \times w}$ 表示N个binary mask的联合（合并到一张图上），表示$L_{route}$ 只在人脸区域内计算。

## 2.3 训练

- 分成两个阶段：
  - single-ID，multi-ID fine-tuning

### Stage 1 : Single ID Training

- 只有ID Embedding Module
- 首先crop and align face region
- 如果训练集中的人脸有多张图，会以0.1的概率使用另外一个 crop and align的人脸，给人脸识别分支提取特征使用（CLIP分支还是使用target的图像）
- 对于CLIP分支，训练时会随机drop：
  - 0.33的概率完全dropCLIP分支
  - 保留CLIP分支，但以0.33的概率drop tokens
  - 完全保留CLIp分支：0.34的概率
- 对UNet使用LoRA
- 使用普通的diffusion loss训练

### Stage 2 : Multiple ID Fine-tuning

- 固定ID embedding模块，只训练LoRA和 multiple id module
- lora的学习率乘以0.1
- loss为扩散loss + $L_{route}$ ，$\lambda = 0.1$ 

# 3 实验

## 3.1 数据

- 数据来源：
  - 240k 单人脸图像，从LAION过滤得到
  - 100k 单人脸图像，从CelebA数据集得到
  - 160k高质量单人脸图像，从网络搜集得到
  - 120k高质量多人脸肖像图像，从LAION过滤得到
- 前3个数据集用于寻来你Stage1
- 最后一个数据集用于训练Stage2
- 使用Qwen-VL打标，LAION的数据使用原始文本描述

## 3.2 实现细节

- 基于SD1.5
- 人脸识别backbone使用CurricularFace
- CLIP使用 OpenCLIP’s clip-vit-huge-patch14
- Q-Former有6层，16 learnable queries
- U-Net的LoRA rank=128
- 8 V100GPU， AdamW，bs=128, lr=1e-5
- stage1 训练了300k iters，stage2 训练了150k iters。
- 5% drop face for uncond training

