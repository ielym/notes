# 0 引用

- [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)



# 1 介绍

- Transfusion可以同时生成离散和连续的模态：
  - 使用50%的文本数据，50%的图像数据
  - next token prediction for text，causal attention
  - diffusion for images, bidirectional attention

# 2 Background

- language modeling loss:

  ![image-20241105234400800](imgs/54-TransFusion/image-20241105234400800.png)

- Diffusion loss : 

  ![image-20241105234420954](imgs/54-TransFusion/image-20241105234420954.png)

- Noise Schedule：

  - 使用cosine scheduler：$\sqrt{\bar{\alpha_t}} \approx cos(\frac{t}{T} \cdot \frac{\pi}{2})$ 

# 3 Transfusion

## 3.1 Data Representation

- 两种数据模态：
  - 离散的文本
  - 连续的图像，使用VAE编码成连续的tokens，拼接BOI和EOI，并插入文本序列中

## 3.2 Model Architecture

- 最主要的参数来源于transformer：

  ![image-20241106002852517](imgs/54-TransFusion/image-20241106002852517.png)

  - 训练了0.16B, 0.37B, 0.76B, 1.4B, 7B几个版本，基于llama

- 输入是 $\mathbb{R}^d$ 的向量：

  - 文本：使用tokenizer输入和输出离散序列（llama2 tokenizer）

  - 图像：尝试了两种方式把 $k \times k$ 的patches转换成向量：

    - 两种方法都是在VAE之后：
      - VAE是自己训练的，又86M参数，CNN结构，latent dimension是8，8倍下采样：256x256 -> 32x32x8. 训练了1M steps

    ![image-20241106001138829](imgs/54-TransFusion/image-20241106001138829.png)

    - 简单的线性层，patch size=2x2
    - U-Net（进一步压缩了token维度，计算量降低了约64x，只损失了很小的性能）

## 3.3 Transfusion Attention

![image-20241106001521241](imgs/54-TransFusion/image-20241106001521241.png)

- 结合使用casual and bidirectional attention：
  - 文本单向
  - 图像双向，使得在同一张图像内部，每个patch都能看到其他patches，但是也只能单向的看到之前的其他图像或文本。
    - intra-image attention能够显著增强模型性能。

## 3.4 Training Objective

- 同时使用语言模型的损失和扩散损失

![image-20241106001621431](imgs/54-TransFusion/image-20241106001621431.png)

## 3.5 Inference

- LM mode：
  - 标准的next token prediction方式
- Diffusion mode : 
  - 当NTP预测到 BOI 时，切换到diffusion mode
  - 初始为纯噪声，并去噪T次
  - 每个去噪阶段，预测 $x_{t-1}$ ，并重新覆盖替换 $x_t$ 继续预测diffusion
  - 当T次去噪循环完成之后，图像覆盖替换之前的位置，并拼接一个 EOI
  - 之后重新切换到 LM mode

# 4 Experiments

## 4.1 Data

- 图像 center-cropped and resized to 256x256
- 随机排序图像和文本，但80%的时间是文本在前

## 4.2 Latent Image Representation

- 在连续VAE之外，还训练了一个VQ-VAE，包含16384个tokens

# 5 Ablations

## 5.1 Attention Masking

![image-20241106003415111](imgs/54-TransFusion/image-20241106003415111.png)

- U-Net比Linear好
- 双向attention比单向好
- 由于U-Net中也使用了双向attention，已经有了信息流动，因此在transformer中使用双向或单向的attention的差别不大。而线性层就很明显。

## 5.2 Patch Size

![image-20241106003810969](imgs/54-TransFusion/image-20241106003810969.png)

- 使用Linear时，patch越多（patch size越小），效果越好

## 5.3 Patch Encoding/Decoding Architecture

![image-20241106004501331](imgs/54-TransFusion/image-20241106004501331.png)

- 使用U-Net比Linear好，但这可能是由于U-Net具有更多参数量导致的
- 为了消除这种影响，实验对比了在不同参数量的transformer下，二者的效果：
  - 随着transformer的参数量增加，unet所占的比例越来越少，和linear的差距也越来越小
  - 但Unet始终都是有优势的

## 5.3 Image Noising

![image-20241106005041074](imgs/54-TransFusion/image-20241106005041074.png)

- 80%的数据是caption在图像之前，图像可以看到文本。
- 实验了对于剩下20%（图像看不到文本，如图像理解任务），加噪步数最大只有 t=500（仍然保持diffusion的objective）。
- 可以看出，使用了 noise limiting策略显著提升了图像理解任务，CIDEr指标（Consensus-based Image Description Evaluation Relevance）