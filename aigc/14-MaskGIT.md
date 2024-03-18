MaskGIT: Masked Generative Image Transformer

# 1 摘要

当前最好的生成式 transformer 模型仍然把图像看作为一系列的 tokens，逐行的根据编码图像序列扫描图像。我们发现这种策略既不是最优的，也不是最高效的。

本文提出了一种新的图像生成范式，使用双向 transformer 编码器，称为 MaskGIT。

在训练中，MaskGIT MaskGIT通过关注所有方向的tokens来学习预测随机掩码的tokens；在推理中，模型模型同时生成图像的所有tokens，之后迭代微调图像。

实验证明，MaskGIT在ImageNet上优于state-of-the-art，并且加速自回归编码 64 倍。

此外，我们证明了 MaskGIT可以被简单的拓展到各种图像编辑任务，如 inpainting, extrapolation和图像编辑。

# 2 介绍

受到 NLP 中 Transformer 和 GPT 成功的启发，生成式 transformer 模型在图像生成领域受到了广泛的关注。通常这些方法的生成过程包含两个阶段：

- 首先把图像量化成一系列 tokens 序列（如使用VQ-VAE）
- 使用自回归模型（如，transformers），基于之前生成的结果（如自编码器的 decoding），来生成图像的 tokens

现在的方法大多数关注第一个阶段，即如何量化图像能够达到最小的信息损失。对于第二个阶段，通常与NLP中的相同。因此，即使是目前最好的生成式 transformer模型仍然把图像当成flattened 为 1D tokens 的序列，之后逐行逐列的按顺序扫描。

我们任务这种表示方式既不是最优的，也不是最高效的。不同于文本，图像本身就不是序列。想象如何创造一个艺术作品：

- 画家从零开始
- 逐步调整填充细节（这与之前的工作中使用的逐行输出的方法形成鲜明对比）

此外，把图像当成一个 flat 的序列，意味着自回归序列的长度随图像分辨率的平方的增长。这不仅给长期相关的建模带来了挑战，而且使解码变得棘手。例如，在GPU上用32x32 的tokens 自动回归生成一张图像需要30秒。

本文提出了一种新的双向transformer （Bidirectional transformer） 用于图像生成，称为 Masked Generative Image Transformer (MaskGIT) ：

- 训练过程中，MaskGIT 使用预测 mask 作为代理任务
- 推理时，MaskGIT 采用一种新的非自回归解码的方法，使用固定的步长来生成图像。在每次迭代中，模型同时并行预测所有的 tokens，但只保留置信度最高的一个。剩余的 tokens 作为被mask掉的部分，会在接下来的迭代中重新进行预测。因此，mask 掉的tokens的比例是捉奸减少的，知道所有的 tokens 都被生成。

如下图所示，相较于逐行逐列扫描生成的方式，MaskGIT的解码速度比其快指数级。因为MaskGIT只迭代8此，而不是256次。

此外，相较于逐行扫描的方法，只能使用当前tokens之前的序列作为条件，bi-transformer 的双向注意力机制能够从已有tokens的各个方向来生成新的tokens。

我们发现，mask的策略（如每次迭代中mask掉的图像的比例）对生成质量又显著影响，我们使用余弦策略，并通过消融实验证明该策略是有效的。

![image-20230624030433206](imgs/14-MaskGIT/image-20230624030433206.png)

在 Imagenet 数据集上，256x256分辨率和512x512分辨率下，我们实验证明 MaskGIT 即快（256x）又好。

此外，MaskGIT多方向生成的特性，使其很自然的能够扩展到图像编辑任务，即使我们的任务没有专门针对这些任务设计，如下图所示：

![image-20230624031628152](imgs/14-MaskGIT/image-20230624031628152.png)

# 3 方法

![image-20230624032957363](imgs/14-MaskGIT/image-20230624032957363.png)

网络整体流程如上图所示，本文的目标是改进上图中的第二个阶段，因此第一个阶段就直接使用VQGAN模型来编码。

对于第二个阶段，我们提出了一种通过 Masked Visual Token Modeling (MVTM) 来学习一个  bi-directional transformer 。

## 3.1 MVTM in Training

