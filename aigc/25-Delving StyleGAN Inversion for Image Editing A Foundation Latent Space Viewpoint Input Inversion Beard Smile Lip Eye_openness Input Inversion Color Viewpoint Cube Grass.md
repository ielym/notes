`Delving StyleGAN Inversion for Image Editing: A Foundation Latent Space Viewpoint Input Inversion Beard Smile Lip Eye_openness Input Inversion Color Viewpoint Cube Grass`

主页：https://kumapowerliu.github.io/CLCAE

StyleGAN中， $W, W^+, F$ 的可编辑能力越来越差，但是重建能力越来越强。最近的方法大多关注基于 $W^+$ 和 $F$ 的图像编辑，以此来提升重建能力，同时保持可编辑能力。

本文提出从隐空间 $W$ 来获得合适的隐编码，方法是通过对比学习的方式对比 $W$ 空间和图像空间，之后使用交叉注意力机制从 $W$ 转换至 $W^+$ 和 $F$ 。实验证明，最基础的 $W$ 空间能够提升表示能力，获得了最好的重建和编辑结果。

# 1 Introduction

GAN inversion的目的是把输入图像映射回隐空间，之后能够进行真实图像编辑。GAN inversion的关键部分是找到 inversion space，避免编辑后的图像失真，同时也要保证具有编辑能力。普遍的 inversion space包括 $W^+$ ，$F$ 。其中，$W^+$ 在防止图像失真和可编辑性上比较平衡，因此许多方法都是在 $W^+$ 上进行编辑。此外，$F$ 包含空间信息。

本文提出了一种 two-step 的方法来改善 $W^+$ 和$F$ 的表示能力：首先获取合适的 $W$ 空间的隐编码；之后使用$W$ 来知道 $W^+$ 和 $F$ 的生成：

- 对于第一个阶段，我们提出了一种对比学习范式来对齐 $W$ 和 图像空间。这种范式受 CLIP 启发，只不过把文本分支替换成 $W$ 分支。我们使用预训练的StyleGAN来构建包含图像$I$ 和其隐编码 $w \in W$ 的成对数据。对比学习中，我们训练两个编码器分别获得 $I$ 和 $w$ 的特征表示，并进行对齐。
- 在GAN Inversion过程中，我们固定对比学习模块，并把该模块当成一个损失函数。该损失的目的是使得图像和其对应的隐编码 $w$ 足够接近。这种方法于现有方法不同，现有方法在图像空间进行对齐，如原始图像和重建图像，然而基于图像空间的对齐没有强制对齐 $W$ 。
- 在找到 合适的隐编码空间$W$之后，我们使用一个交叉注意力编码器，用于把 $w$ 转换成 $w^+ \in W^+$ 和 $f \in F$ ：
  - 当计算 $w^+$ 时，我们把 $w$ 作为 $Q$ ，把 $w^+$ 作为 $K, V$ 来重建 $w^+$ 。这种方法能够强制使得 $w^+$ 接近 $Q$ ，使得 $w^+$ 能够具有类似 $w$ 的可编辑能力。
  - 当计算$f$时，把 $w$ 当作 $K, V$ ，把 $f$ 当作 $Q$ 。因此 $w$ 会知道$f$ 进行特征重整。
- 最终，使用 $w^+$ 和 $f$ 来生成重建结果。

本文方法称为 Contrastive Learning and Cross-Attention Encoder, CLCAE 

# 2 Related work

## Latent Space Editing

图像编辑可分为两大类：有监督和无监督

- 有监督：
  - InterfaceGAN使用有标注的图像来训练一个二分类的SVM，并找到编辑方向
- 无监督：
  - GanSpace通过PCA来找到编辑方向
  - [Feat:  Face editing with attention], [Styleclip: Text-driven manipulation ofstylegan imagery], [Hair-clip:  Design your hair by text and reference image], [One model to edit themall: Free-form text-driven image manipulation with semanticmodulations] 等方法使用CLIP损失[Learn-ing transferable visual models from natural language super-vision] 实现图像编辑。
  - 其他一些方法使用inpaint的方法进行图像编辑 [Human motionformer:  Transferringhuman  motions  with  vision  transformers], [Rethinking image inpainting via a mutual encoder-decoder  with  feature  equalizations], [Coherent semantic attention for image inpainting], [Pd-gan: Probabilistic diverse gan for im-age inpainting], [Deflocnet: Deep im-age editing via flexible low-level controls]

本文使用InterfaceGAN和GanSpace来找到语义方向，并验证编辑的效果。

# 3 Method

![image-20230815212716603](imgs/25-Delving%20StyleGAN%20Inversion%20for%20Image%20Editing%20A%20Foundation%20Latent%20Space%20Viewpoint%20Input%20Inversion%20Beard%20Smile%20Lip%20Eye_openness%20Input%20Inversion%20Color%20Viewpoint%20Cube%20Grass/image-20230815212716603.png)

- CNN Encoder来自于pSp，
- 对于输入图像 $I$ ，我们获得隐编码 $w \in W \in \mathbb{R}^{512}$ 
- $W$ 空间和图像空间通过对比学习进行对齐
- 之后，把 $w$ 作为 $Q$ ，通过一个交叉注意力模块来获取 $w^+ \in W^+ \in \mathbb{R}^{N \times 512}$ ，$N$ 与生成模型有关。如，当生成尺寸是1024时，$N = 18$ 

- 此外，我们选择 Encoder的 $T_3$ 特征作为 $f \in F \in \mathbb{R}^{H \times W \times C}$ ，并使用 $w$ 作为 $K, V$ 来重整 $f$ 。
- 最后，把 $w^+, f$ 送到 预训练好的 StyleGAN中来获得重建的结果。

## 3.1 Aligning Images and Latent Codes

我们使用CLIP中的对比学习来对齐图像 $I$ 和隐编码 $w$ 。当该部分预训练好之后，我们冻结该部分，并将其作为一个损失函数来度量图像和隐编码的相似度，该loss用于训练 CNN encoder，来对齐上图所示的 Input 和 $w$ 。

**NOTE：**先有鸡还是先有蛋的问题。训练 CLIP 时，使用的是 StyleGAN 随机生成的 $w$ 和图像，此时还没有 CNN encoder 。

![image-20230815214144725](imgs/25-Delving%20StyleGAN%20Inversion%20for%20Image%20Editing%20A%20Foundation%20Latent%20Space%20Viewpoint%20Input%20Inversion%20Beard%20Smile%20Lip%20Eye_openness%20Input%20Inversion%20Color%20Viewpoint%20Cube%20Grass/image-20230815214144725.png)

对比学习模块如上图所示，我们用StyleGAN合成 100K 个和 $I - w$ 数据对，该模块包含两个 Encoder，一个Image Encoder （CNN）用于编码图像，一个 Latent code Encoder （Transformer） 用于编码 $w$ 。

假设一个batch包含 $S$ 个图像和隐编码对 $I \in \mathbb{R}^{256 \times 256 \times 3}, w \in \mathbb{R}^{512}$ 。经过各自的Encoder之后，把编码特征分别定义为 $h_I(I) \in \mathbb{R}^{512}, h_w(w) \in \mathbb{R}^{512}$ 。对比学习的loss为：
$$
L_i^{I \to w} = -log 

\frac

{exp [<h_I(I_i), h_w(w_i)> / t]}

{\sum_{k=1}^{S} exp [<h_I(I_i), h_w(w_k)> / t]}
$$

$$
L_i^{w \to I} = -log 

\frac

{exp [<h_w(w_i), h_I(I_i)> / t]}

{\sum_{k=1}^{S} exp [<h_w(w_i), h_I(I_k)> / t]}
$$

其中，$<\cdot>$ 表示余弦相似度，$t \in \mathbb{R}^+$ 是一个可学习的温度超参。

最终的loss为：
$$
L_{align} = 

\frac{1}{S}

\sum_{i=1}^S

(\lambda L_i^{(I \to w)} + (1 - \lambda) L_i^{(w \to I)})
$$
 其中，$\lambda = 0.5$ 。

本文把 pSp 的CNN部分作为 $h_I(\cdot)$ ，把 StyleTransformer作为$h_w(\cdot)$ 。

当对比学习模块训练好之后，我们把该模块冻结，并把 $L_{align}$ 作为 CNN Encoder的损失。

## 3.2 Cross-Attention Encoder

一旦对比学习模块训练好之后，我们冻结其参数，并将其作为 image 和 latent code的匹配损失。该损失用于训练 CNN Encoder。CNN Encoder是一个金字塔结构，来自于pSp，产生多级特征（$T_1, T_2, T_3$） ，我们使用 $T_1$ 并通过 pSp 中的 map2style block 来产生隐编码 $w$ 。

在获取到 $w$ 之后，我们可以可以使用 $I, w$ 来计算对齐损失。

### 3.2.1 $W^+$ Cross-Attention Block

 ![image-20230815215851887](imgs/25-Delving%20StyleGAN%20Inversion%20for%20Image%20Editing%20A%20Foundation%20Latent%20Space%20Viewpoint%20Input%20Inversion%20Beard%20Smile%20Lip%20Eye_openness%20Input%20Inversion%20Color%20Viewpoint%20Cube%20Grass/image-20230815215851887.png)

- 首先，使用map2style获取粗粒度的 $\Delta w^+ \in \mathbb{R}^{N \times 512}$ 
- 之后，把 $\Delta w^+$ 中的每个向量 $\Delta w_i^+ \in \mathbb{R}^{512}$ ，以及 $w \in \mathbb{R}^{512}$ 送到 $w^+$ Cross-Attention Block，来预测更好的 $\Delta w_i$ ，其中 $i = 1,...,N$ 。
- 在交叉注意力模块中，我们把 $w$ 作为 $Q$，把 $\Delta w_i$ 作为 $K, V$ 来计算attention map。可以使得 $w^+$ 更接近 $w$ 。
- 最后，把 $w$ 和 $Attention(Q, K, V)$ 通过残差链接来获得最终的 $w_i^+$ 

整个流程可以表示为：
$$
Q = w W_Q^{w^+} \\
K = \Delta w_i^+ W_K^{w^+} \\
V = \Delta w_i^+ W_V^{w^+} \\
Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt(d)})V \\
w_i^+ = w + Attention(Q, K, V)
$$
其中，$W_Q^{w^+}, W_K^{w^+}, W_V^{w^+} \in \mathbb{R}^{512 \times 512}$  。

- 交叉注意力中使用了多头注意力机制。

- 由于保证了 $w^+$ 接近 $w$ ，因此 $w^+$ 具有极强的编辑能力。
- 此外，重建性能也不会损失太多，因为我们使用$L_{align}$强制保证 $w$ 和原始图像对齐了

### 3.2.2 F Cross-Attention Block

![image-20230815220828325](imgs/25-Delving%20StyleGAN%20Inversion%20for%20Image%20Editing%20A%20Foundation%20Latent%20Space%20Viewpoint%20Input%20Inversion%20Beard%20Smile%20Lip%20Eye_openness%20Input%20Inversion%20Color%20Viewpoint%20Cube%20Grass/image-20230815220828325.png)

丰富且正确的空间信息可以提升 $f$ 的表示能力。由于 $T_3$ 在CNN金字塔中有最丰富的空间信息，我们使用 $T_3 \in \mathbb{R}^{64 \times 64 \times 512}$ 作为基础的特征来预测 $f$ 。

之后我们在 $w$ 和 $T_3$ 之间计算交叉注意力，并通过残差来重整 $T_3$ 。

与 $W^+$ 不同的是，我们把 $w$ 作为 $V, K$ ，把 $T_3$ 作为 $Q$ ，这是因为我们想挖掘 $w$ 中的空间信息来支持 $T_3$ 。

最后，我们使用一个 CNN 来减小空间尺寸，并获得最终的 $f$ 。$f$ 和特征空间$F$ 的尺寸相同。

整个流程为：
$$
Q = T_3 W_Q^f \\
K = w W_K^f \\
V = w W_V^f \\
Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d}})V \\
f = CNN[Attention(Q, K, V) + T_3]
$$
其中，$W_Q^f, W_K^f, W_V^f \in \mathbb{R}^{512 \times 512}$ 。

## 3.3 Image Editing

编辑阶段，我们需要获得修改后的 $\hat{w}^+$ 和 $\hat{f}$ ：

- 对于 $\hat{w}^+$ ，我们是哟个经典的隐空间编辑方法 [Ganspace: Discovering interpretable gan con-trols] 和 [Interpreting the latent space of gans for semantic face editing]

- 对于 $\hat{f}$ ：

  - 首先按照 FS [A style-based gan encoder for high fidelity reconstruc-tion of images and videos] 来生成重建的图像 $G(w^+)$ 和 $G(\hat{w^+})$ ，$G(\cdot)$ 是生成器。

  - 之后分别提取  $G(w^+)$ 和 $G(\hat{w^+})$ 的第5个卷积层的特征。

  - 最后，计算两个的差异性，并加到 $f$ 中来获得 $\hat{f}$ 

  - 整个计算流程可以表示为：
    $$
    \hat{f} = f + G^5(\hat{w}^+) - G^5(w^+)
    $$

通过 $\hat{w}^+, \hat{f}$ ，最后计算最终的编辑结果 $G(\hat{w}^+, \hat{f})$



## 3.4 损失函数

为了训练encoder，我们使用通用的 ID 和 重建损失来同时优化三个重建结果：$I_{rec}^1 = G(w), I_{rec}^2 = G(w^+), I_{rec}^3 = G(w^+, f)$  。此外我们使用特征正则化来使得  $f$ 接近原始的特征$G$ 

### 3.4.1 Reconstruction losses

我们使用 pixel-wise L2 loss 和 $L_{LPIPS}$ 来度量输入图像和重建图像的像素级别和感知级别的相似度：
$$
L_{rec} = \sum_{i=1}^3 
(\lambda_{LPIPS} L_{LPIPS}(I, I_{rec}^{i}) + \lambda_2 L_2 (I, I_{rec}^{i}))
$$
其中，$\lambda_{LPIPS} = 0.2, L_2 = 1$ 

### 3.4.2 ID loss

我们按照 e4e 中的方法来使用 identity loss
$$
L_{id} = \sum_{i=1}^3 (1 - < R(I), R(I_{rec}^{i}) > )
$$
对于人脸数据，$R$ 是预训练的 ArcFace人脸识别网络。对于汽车数据，$R$ 是ResNet-50（在MOCOv2训练）

### 3.4.3 Feature regularization

对于 $\hat{f} = f + G^5(\hat{w}^+) - G^5(w^+)$ ，我们需要约束其与原始的 $G^5(w^+)$ 相似：
$$
L_{f_{reg}} = || f - G^5(w^+) ||_2^2
$$

### 3.4.4 Total losses

$$
L_{f_{reg}} = 
\lambda_{rec} L_{rec}
+
\lambda_{ID} L_{ID}
+
\lambda_{f_{reg}} L_{f_{reg}}
+
\lambda_{align} L_{align}
$$

其中，$\lambda_{rec} = 1, \lambda_{ID} = 0.1, \lambda_{f_{reg}} = 1$

# 4 Experiments

## 4.1 Implementation Details

- 对比学习：使用Adam优化器训练图像和隐编码 encoders。bs=256
- StyleGAN inverse阶段，使用FFHQ训练，CelebA-HQ测试。输入分辨率 256，Ranger优化器，bs=32，8-V100 GPUs

