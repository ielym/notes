`PhotoVerse Tuning-Free Image Customization with Text-to-Image Diffusion Models`

# 1 介绍

- 提出PhotoVerse，在文本和图像领域使用双分支结构。
- 此外，引入人脸身份损失，在训练时增强身份不变。
- 推理时无需训练

# 2 方法

## 2.1 双分支概念提取

- 在提取人脸特征前，非常有必要预处理输入图像：
  - 人脸检测算法来检测人脸，保证后续特征提取只关注人脸区域
  - 检测框外扩，如 1.3x，保证人脸区域都被包含在crop的图像中
  - 为保证生成模型的输入尺寸，需要缩放crop图像
  - 为了进一步移除非人脸信息，需要对resize的图像做分割

## 2.2 文本条件

文中说参考了：`Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation` 和 `An image is worth one word: Personalizing text-to-image generation using textual inversion`

- 使用CLIP的image encoder来提取图像特征，并使用 `Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation` 的方法来增强image tokens的表达能力：选择CLIP image encoder的 m个layers的特征，能够在不同levels的抽象概念上获得空间和语义信息。
- 随后，一个 multi-adapter结构用来把 m 个不同层上提取的特征转换成 m 个文本编码维度的特征 $S^* = S_1, ..., S_m$ 。
- 由于CLIP能够有效的对齐文本和图像编码，因此上一条说的 multi-adapter 中的每个 adapter 都是一个只有两层的 MLP，负责把输入特征映射到1024-d的维度。每个全连接后都有一个LN + Leaky-ReLU。

## 2.3 视觉条件

尽管从人脸上提取到了text embedding空间的特征，但仍有局限：

- 效果可能被接下来处理text embedding的text encoder所影响
- 文本编码太抽象，需要更强大的表达能力才可以

因此还需要从图像空间作为辅助，更加自然，也更加便于理解新的概念。

为了提取视觉编码：

- 使用从CLIP image encoder获取的特征
- 使用和 text adapter中相同结构的image adapter来映射特征

## 2.4 双分支概念注入

由于微调UNet太昂贵，并且可能降低模型的可编辑能力（过拟合导致），因此本文只在cross-attention模块中微调（E4T和Custom Diffusion也已经证明了attention层是最有表达能力的）

![image-20231008233413946](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008233413946.png)

如上图所示：

- 对于文本分支：

  - 使用如 LoRA等高效的方法进行微调，加入LoRA后的推理过程可以表示为：

    ![image-20231008233602879](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008233602879.png)

    其中，$p$ 表示text encoder的特征，$\alpha$ 是lora的权重，$\Delta W =BA$ 表示LoRA的低秩分解。

- 对于视觉分支：

  - 由于文本分支是SD本来就有的，因此可以使用LoRA微调。而SD之前没有 visual cross-attention分支，因此需要重新增加该网络结构并重新训练，因此不再使用LoRA：
    $$
    K^S = W_k^Sf 
    \\
    V^S = W_v^Sf 
    $$
    其中，$f$ 是从image adapter提取的特征。

- 双分支融合：

  ![image-20231008234027221](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008234027221.png)

  其中，$\gamma, \sigma$ 分别表示两个分支的缩放系数。训练时为了增强模型推理的鲁棒性，训练时动态的设置 $\gamma, \sigma$ ：

  - 从 $U = (0, 1)$ 的均匀分布中采样一个随机数 seed
  - 如果 $seed < r_1$ 时，$\gamma$ 设置成2，且 $O = \gamma Attn(Q, K^T, V^T)$ 
  - 如果 $seed < r_2$ 时，$\sigma$ 设置成2，且 $O = \sigma Attn(Q, K^S, V^S)$ 
  - 否则，$\gamma = \sigma = 1$ ，$O = \gamma Attn(Q, K^T, V^T) + \sigma Attn(Q, K^S, V^S)$

  其中，$r_1, r_2$ 是 seed的两个阈值。

## 2.5 损失

## 2.5.1 正则损失

不知道为何这样做

- 对 text adapter输出的特征 $p_f$ 计算正则：

  ![image-20231008235150493](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008235150493.png)

- 对图像分支的 $V^S$ 计算正则：

  ![image-20231008235228653](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008235228653.png)

  

### 2.5.2 感知损失

对推理的图像 $x$ 和 `a phota of S^*` 推理的图像 $x'$，使用ArcFace提取特征 $f(x), f(x')$，并计算余弦相似度 $C(\cdot)$ ：

![image-20231008235424819](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008235424819.png)

### 2.5.3 扩散模型损失

和正常训练SD的相同，$L_{diffusion}$

### 2.5.4 总损失

![image-20231008235516929](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231008235516929.png)

几个 $\lambda$ 是权重。

# 3 实验

## 3.1 数据集

在3个公开数据集上微调模型：

- Fairface
- CelebA-HQ
- FFHQ

验证集是自己构建的，包含326张自己搜集的图像，包含不同种族（白，黑，棕，黄）和性别。

推理时，使用 `a photo of S^*` 来生成图像。

## 3.2 实现细节

- 使用SD作为底模来微调LoRA，visual attention，和 adapters 。
- 学习率 1e-4，bs 64，V100 GPU
- 训练60000个step
- 训练时 $\alpha = 1$, $\lambda_{face} = 0.01, \lambda_{rt} = 0.01, \lambda_{rv} = 0.001, m = 5, r_1 = 1/3, r_2 = 2/3$ 
- 推理时 $m = 1, \alpha = 1$ ，也可以调整 $m, \sigma, \gamma, \alpha$ 来获得更灵活的推理。
- 推理时也可以使用图生图方法。

# 4 结果

![image-20231009000552932](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231009000552932.png)

此外，该方法也能够修改发型等属性：

![image-20231009000620583](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231009000620583.png)

## 4.1 消融实验

视觉条件和3个loss的消融实验：

![image-20231009000716005](imgs/32-PhotoVerse%20Tuning-Free%20Image%20Customization%20with%20Text-to-Image%20Diffusion%20Model/image-20231009000716005.png)
