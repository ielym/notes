# 0 引用

- [Show-o: One Single Transformer to Unify Multimodal Understanding and Generation](https://arxiv.org/abs/2408.12528)

# 1 介绍

- 统一的transformer架构用于多模态理解和生成。
- 不是全部都自回归，结合了自回归和离散的扩散模型来处理不同模态的输入和输出。
- 支持众多人物，包括VQA, T2I，text-guided inpainting/extrapolation, and mixed-modality generation

---

![image-20241111233355426](imgs/55-show-o/image-20241111233355426.png)

- 如上图 c 所示，目前的多模态理解和生成模型通常分开处理不同模态。
- 文本使用自回归方式进行生成是make scene的，但是图像用自回归的方式生成是否是最优的？
  - 由于casual attention，next token 的预测方式需要巨大的采样次数。
  - 扩散模型使用full attention的生成能力更好。

---

因此，本文的motivation就是用一个transformer，同时进行自回归和扩散建模：

- 同时处理离散和连续tokens不容易
- 目前最好的扩散模型通常都需要两个独立的模型（text encoder, denosing network），只用一个transormer进行图像生成也存在挑战。

show-o:

- 基于预训练的LLM模型
- 使用离散的image token，一个text tokenizer编码文本，不需要额外的text encoder
- 提出统一个prompting策略来把tokens处理成结构化的序列作为输入

# 2 方法

![image-20241112002938068](imgs/55-show-o/image-20241112002938068.png)

## 2.1 Tokenization

![image-20241112001024078](imgs/55-show-o/image-20241112001024078.png)

- **Text Tokenization :** 使用预训练的LLM，并使用相同的tokenizer，没有任何改动。
- **Image Tokenization :**  使用35M图像数据，根据MAGVIT-v2训练一个lookup-free quantizer，codebook size K = 8192。把 256x256编码成16x16的离散tokens，如上图（c）所示。

---

- 除了离散的token之外，还实验了连续token，来自于 MAGVIT-v2和CLIP-ViT。分别如上图 (b), (c) 所示。
- 后续实验默认使用离散token进行实验。

## 2.2 结构

- 使用预训练的LLM （phi-1.5 或 Llama），每个attention layer都新增了QK-Norm，除此之外没有其他改动。
- 由于是基于LLM，因此不需要额外的text encoder。

### 2.2.1 Unified Prompting

![image-20241112002710425](imgs/55-show-o/image-20241112002710425.png)

- Unified Prompting策略用于统一输入数据。
- 对于图像-文本对 $(x, y)$ ：
  - tokenized成为 M 个图像tokens $u = \{ u_i \}_{i=1}^M$ ，和 N 个文本 tokens $v = \{ v_i \}_{i=1}^N$ 

- 不同任务的token组合形式如上图所示，
  - [MMU] , [T2I] ：预定义的 task tokens
  - [SOT], [EOT] ：文本token的起止tokens
  - [SOI], [EOI] ：图像token的起止toknes

### 2.2.2 Omni-Attention Mechanism

![image-20241112003247118](imgs/55-show-o/image-20241112003247118.png)

- 根据不同的输入序列使用不同的attention机制
- text tokens内部使用causal attention
- image tokens 内部使用full attention
- 多模态图像理解任务：文本可以看到之前所有的图像tokens
- 文生图任务：图像可以看到之前所有的文本tokens
- 只有文本时：原始的causal attention

### 2.2.3 Training Objectives

- 包含两种训练目标：

  - Next Token Prediction：用于文本生成

    ![image-20241112003704235](imgs/55-show-o/image-20241112003704235.png)

  - Mask Token Prediction：用于图像任务，方法和 MaskGIT 相同。不同时间步有不同的tokens被mask掉，使用 [MASK] 表示。

    ![image-20241112004401974](imgs/55-show-o/image-20241112004401974.png)

- 最终的loss为二者相加：

  ![image-20241112004441264](imgs/55-show-o/image-20241112004441264.png)

## 2.3 训练

使用了三个阶段的渐进式训练策略来解决图像和文本对齐的问题：

- **Image Token Embedding and Pixel Dependency Learning** ：
  - 数据：
    - RefinedWeb保持语言模型的能力
    - Imagenet-1k用于class-conditional图像生成
    - 图文数据对用于image caption任务
  - 对于图像生成：
    - 直接使用imagenet-1k的类别名称作为文本输入，来学习class-conditional的图像生成。
  - 该阶段主要有三个目标：
    - 训练图像中新增的可学习embeddings
    - 学习图像生成中的像素依赖
    - 对齐image caption中的图像和文本模态
- **Image-Text Alignment for Multimodal Understanding and Generation** :
  - 基于第一阶段的预训练权重，使用图文数据对替代imagenet-1k。
  - 该阶段主要关注同时对齐 t2i 和 image caption的图文模态
- **High-Quality Data Fine-tuning:**
  - 使用高质量的图文数据对，以及指令数据。用于多模态理解和混合模态生成。

---

超参：

- 训练步数：
  - 第一个阶段训练500K steps
  - 第二个阶段额外训练 1000K steps
- 48 A100 80G
- bs = 1152
- AdamW, weight decay = 0.01， 5000 steps for warm-up
- 初始学习率1e-4，余弦衰减策略

## 2.4 推理

- Multimodal understanding:
  - 给定图像和问题，自回归生成text tokens
- Visual generation:
  - 初始时提供N个文本tokens和M个 [MASK] tokens作为输入
  - 使用T个时间步进行图像生成。

# 3 实验

## 3.1 消融实验

### 3.1.1 多模态理解的数据集规模和图像分辨率

![image-20241112011010409](imgs/55-show-o/image-20241112011010409.png)

由于图像token在预训练LLM中没有见过，需要从头训练。

- 增加图像分辨率和数据规模都能提高性能。

### 3.1.2 多模态理解的Vision Encoder

![image-20241112011150838](imgs/55-show-o/image-20241112011150838.png)

- CLIP最好的原因：已经经过了大规模数据的预训练；MAGVIT是重建损失，而CLIP是对比损失

### 3.1.3 多模态理解的数据表示

- 仍然如3.1.2的table 5所示，使用连续tokens也能提高性能。
- 作者认为是由于常见的多模态理解训练数据，如llava-pretrain-558K规模较小，不足以把离散token对齐到文本空间。
- 而连续token更容易对齐
