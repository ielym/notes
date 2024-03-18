`StyleDiffusion: Prompt-Embedding Inversion for Text-Based Editing`

# 1 Motivation

现有图像编辑方法存在两个问题：

- 选中区域内的编辑效果不好，非选中的区域的内容也发生了变化。
- 需要仔细的调整文本描述

为了解决上述问题，本文做了两个改进：

- 只优化 Cross-attention 中的 $W_V$ ，已经足够来重建图像。
- 提出注意力正则化来防止过度的结构改变。



# 2 StyleDiffusion

![image-20231023202330892](imgs/28-StyleDiffusion%20Prompt-Embedding%20Inversion%20for%20Text-Based%20Editing/image-20231023202330892.png)

给定一张图，我们的目标是从一个frozen的预训练模型中获得更加准确的编辑能力。