# 0 引用

[OminiControl: Minimal and Universal Control for Diffusion Transformer](https://arxiv.org/abs/2411.15098)



# 1 介绍

- 基于DiT，使DiT能够使用自身的mm-attn来编码图像conditions：
  - 有效的和高效的聚合图像条件，只增加了0.1%的额外参数
  - 使用统一的方式解决了广泛的image conditioning tasks，包括subject-driven，spatially-aligned的任务

- 使用VAE编码条件图像，并增加了可学习的position embeddings，和latent noise进行拼接。从而可以直接的进行多模态的注意力交互。
- 基于FLUX.1-dev实现。

# 3 方法

## 3.1 定义

- noisy image tokens： $X \in \mathbb{R}^{N \times d}$ ，$N$ 是图像token数量，$d$ 是特征维度。

- text condition tokens ：$C_T \in \mathbb{R}^{M \times d}$  ，$M$ 是文本token数量，$d$ 是特征维度。

- DiT中的MM-Attn（MMA）使用RoPE来编码空间信息：

  - $X_{i,j} \to X_{i,j} \cdot R(i,j)$ 

  - $R(i,j)$ 表示rope位置编码矩阵，$i,j$ 表示空间位置。
  - 在FLUX.1中，文本的位置编码统一设置为 $(0, 0)$ 

- MM-Attention ： 
  - $MMA([X; C_T]) = softmax(\frac{Q K^T}{\sqrt{d}}) V$ 

## 3.2 OminiControl

![image-20241127004527778](imgs/62-OminiControl/image-20241127004527778.png)

### 3.2.1 基于现有的结构

- 使用VAE编码condition tokens $C_I$ ，与噪声图像 $X$ 共享latent space，从而能够直接被transformer block处理。只需要fintune lora来处理额外的condition。

### 3.2.2 统一的序列处理

- 之前的controlnet通过加法的方式进行处理 $X \leftarrow X + C_I$ ，存在局限性：

  - 对于非空间align的任务缺乏灵活性（分辨率不同）
  - 交互不足

- OminiControl直接concat条件 $[X; C_T; C_I]$ 进行多模态处理。可以直接进行多模态的交互。

  - 如下图所示，这种方法可以同时处理spatially aligned和non-aligned 任务：
    - 对于 spatially align 的任务，能够实现更好的空间对齐（对角线的注意力分数更大）
    - 对于non-spatially align的任务，也能够有准确的双向注意力响应区域。

  ![image-20241127004819534](imgs/62-OminiControl/image-20241127004819534.png)

  - 此外，这种方法的训练loss也更低：

    ![image-20241127004602768](imgs/62-OminiControl/image-20241127004602768.png)

### 3.2.3 位置感知的token交互

- RoPE中，以512为例，经过VAE之后的latent为 $32 \times 32$ ，$R(i,j)$ 中的 $i,j \in [0, 31]$ ；$C_T$ 固定是 $(0, 0)$ 。

- 在OminiControl中，对空间对齐 和 非空间对齐的任务使用不同的pe：

  ![image-20241225234605985](imgs/62-OminiControl/image-20241225234605985.png)

  - $\Delta$ 是 offset
  - 对于空间对齐的任务，参考图的位置编码与噪声图像相同
  - 对于非空间对齐的任务，使用offset保证**没有空间overlap** 。如上图 4b 所示，这种方法的loss更低。

### 3.2.4 可控的条件强度

- 使用attn map来控制注入强度：

  ![image-20241225234948025](imgs/62-OminiControl/image-20241225234948025.png)

## 3.3 Subject200K datasets

![image-20241225235113298](imgs/62-OminiControl/image-20241225235113298.png)

- 使用GPT-4o生成超过3w条不同的主体描述，每个描述包含同一个主体在不同场景。

- 把生成的主体描述组合成格式化的结构，每个prompt包含同一个主体在两个不同的场景，模板为：

  ![image-20241226000247654](imgs/62-OminiControl/image-20241226000247654.png)

- 最后，使用GPT-4o来验证是否保持的主体一致和较高的图像质量

# 4 实验

## 4.1 Setup

- 使用LoRA，rank=4
- bs=1, 梯度累计为8，Prodigy optimizer with safeguard warmup and bias correction enabled。
- weight decay 0.01
- 2 H00 GPUs （80G）
- spatial aligned 任务训练5w iters；subject-driven 任务寻来你1.5w iters

![image-20241226000721520](imgs/62-OminiControl/image-20241226000721520.png)

![image-20241226000730427](imgs/62-OminiControl/image-20241226000730427.png)