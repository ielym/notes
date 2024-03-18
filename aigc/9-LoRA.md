LoRA: Low-Rank Adaptation of Large Language Models

- 论文：https://arxiv.org/pdf/2106.09685.pdf

# 1 介绍

提出了一种 **Lo**w-**R**ank **A**daptation (LoRA) ，通过冻结预训练好的模型权重来注入可训练的低秩分解矩阵到Transformer的每一层，极大降低了下游任务的可训练参数量。与使用Adam fine-tuned 的GPT-3 175B相比，LoRA可以降低10000倍的训练参数和3倍的显存。LoRA的质量持平或超过多个fine-tune模型。



# 2 方法

## 2.1 问题描述

假设有一个预训练好的模型 $P_\Phi (y|x)$ ，参数为 $\Phi$ 。当想要把预训练模型应用到下游任务中时，需要在下游数据上进行微调，数据集为  $Z = \{ (x_i, y_i) \}_{i=1, ..., N}$ 。

在微调过程中，模型初始化参数为 $\Phi_0$ ，训练完成之后，模型权重为 $\Phi_0 + \Delta \Phi$ ，其中$\Delta \Phi$ 整个训练过程中的累积的权重。

**上述微调方法的主要缺点为**：对于每个下游任务，都需要学习一个不同的参数 $\Delta \Phi$ ，并且 $\Phi$ 的维度和 $\Delta \Phi$ 相同。因此，如果预训练模型非常大，给每个下游任务都分别训练或存储一整个模型是非常有挑战的。

本文提出使用低秩表达来编码 $\Delta \Phi$ ，当微调 GPT-3 175B时，需要训练的参数量只有 $\Phi$ 的 $0.01\%$ ，非常计算和内存高效。

## 2.2 Low-Rank-Parametrized Update Matrices

作者观察到： Aghajanyan et al.(2020) 证明，预训练的语言模型权重具有较低的“内在维度” (Instrisic dimension)，即使随机映射到较小的子空间，仍然可以有效地学习。

受此启发，作者假设权重更新同样有较低的 Instrisic dimension 。因此：

- 对于预训练的 $W_0 \in \mathbb{R}^{d \times k}$ ，累积梯度为 $\Delta W$ ，$\Delta W$ 进行低秩分解，分解为 $BA$ 。即：
  $$
  W_0 + \Delta W = W_0 + BA 
  $$
  

  其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ ，并且 $r << min(d, k)$ 。

- 在训练过程中， $W_0$ 不进行任何更新，$A, B$ 包含可训练权重。

- 对于输入 $x$ ，原始模型的输出为 $W_0 x$ ，微调之后的输出为 $(W_0 + \Delta W)x = W_0 x + \Delta W x = W_0x + BAx$ 。如下图所示：

![image-20230614163924159](imgs/9-LoRA/image-20230614163924159.png)

- $A$ 使用随机高斯初始化，$B$ 全部初始化为0。因此，最开始微调时，$\Delta W = BA = 0$ ，不会改变预训练模型的输出。 