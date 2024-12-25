[Group Diffusion Transformers are Unsupervised Multitask Learners](https://arxiv.org/abs/2410.15027)

# 1 介绍

- 同时生成一组相关的图像
- 通过concat不同图像的 self-attention tokens实现，能够捕捉到跨图像的关系（身份，风格，布局，环境，颜色等）

# 2 方法

## 2.1 问题定义

- 目标时生成 $n$ 张图 $x = \{ x_1, x_2, ..., x_n \}$ 
- 每张生成图像都有一个文本描述 $c = \{ c_1, c_2, ..., c_n \}$ 
- 可选的，可以设置 $0 \le m \lt n$ 个 $x$ 可以作为参考图像，生成剩余的 $n - m$ 张图像
- 可以支持多种任务：
  - **T2I** ：$n = 1$ ，$m = 0$ 
  - **Font Generation** : $n > 1$, $m = 0$ 
  - **Picture Book Generation** : $n \gt 1$ ，$m = 0$ 
  - **Identity Preservation** ：$n > 1$ ，$m = 0$  。身份信息由文本控制，如，名字或其他身份。
  - **Local Editing** : $n = 2$ ，$m = 1$ 。提供一张参考图像，模型生成编辑后的图像。
  - **Image Translation** : $n = 2$, $m = 1$ 
  - **Subject Customization** ：$(n - m) \ge 1$ ，$1 \le m \lt n$ 个角色图像作为参考图
  - **Style Preservation** ：$(n - m) \ge 1$ ，$m = 1$ 表示参考风格图像。

## 2.2 模型结构

- 问题的关键是在生成时建立不同图像之间的联系。我们做了一个直接的修改：

  - 在self-attn中，concat不同图像的tokens

- 对于不同类型的文本条件的视觉生成架构，需要做一些小的适配：

  - **Encoder-Decoder ：**

    ![image-20241122001741466](imgs/61-Group%20diffusion%20transformers%20are%20unsupervised%20multitask%20learners/image-20241122001741466.png)

    - 类似于PixArt，每一个transformer block包含一个self-attn，以及用于图文交互的cross-attn，和一个FFN。
    - 在self-attn中concat所有image tokens。每张图可以看到其他所有图。
    - self-attn之后，会重新split回去。
    - 在cross-attn中，每张图只能看到自己的文本。

  - Encoder-Only ：

    ![image-20241122002044329](imgs/61-Group%20diffusion%20transformers%20are%20unsupervised%20multitask%20learners/image-20241122002044329.png)

    - 类似于SD3和FLUX，只有self-attn和FFN
    - 把self-attn修改成masked version：
      - 首先，$x_i, c_i$ concat起来（x_mm） 
      - mask：
        - 每个图像token可以看到其他所有的图像tokens
        - 每个文本token只能看到对应的图像token以及文本自身

## 2.3 训练数据

- 关注图像相关的任务，需要高质量，大规模和多样的成组的图像数据集。

- 现有数据集不符合要求，自己构建了一个，构建的方式包括几个关键步骤：

  - 搜集一个足够大量的多模态数据，把图像抽取出来，并保持原始的图像顺序

  - 从中人工标注一个小的子集，把合适分组的作为正样本，不适合的作为负样本

  - 使用标注的数据训练一个二分类器，来过滤搜集的数据

  - 组内和组件去重

  - 最终得到了50k的图像组，分布如下：

    ![image-20241122003554520](imgs/61-Group%20diffusion%20transformers%20are%20unsupervised%20multitask%20learners/image-20241122003554520.png)

- 另外一个关键的步骤是生成准确的描述，来获得组内的相关性：

  - 使用内部的MLLM，迭代着测试最优的prompt，来保证对不同类型的成组图像生成合适的描述，如下图所示：

    ![image-20241122003732994](imgs/61-Group%20diffusion%20transformers%20are%20unsupervised%20multitask%20learners/image-20241122003732994.png)

## 2.4 训练过程

- 使用预训练权重初始化，如pixArt，SD3。GDT没有引入额外的参数。
- 在预训练和FFT阶段，均匀的从1-4之间采样group size，并动态调整bs来保持GPU利用率。
- 预训练约 10w steps，并在一个高质量数据上微调约5000steps
- 使用A100GPUs

## 2.5 推理

![image-20241127000242304](imgs/61-Group%20diffusion%20transformers%20are%20unsupervised%20multitask%20learners/image-20241127000242304.png)

- 由于推理时需要一组prompt，不友好，因此构建了一个用户接口。如上图所示，由两种pipeline：
  - `[Instruction] -> [Group Prompts] -> [Generated Images]` ：用于成组图像生成
  - `[IMGs] -> [Instruction] -> [Group Prompts] -> [Generated Images]` ：用于有条件的成组图像生成

- 使用MLLMs来把用户的指令转换成为成组的prompts：
  - MLLMs分析group size以及对应的任务。如：
    - 假设instruction为 "Draw a line sketch of a female character and the corresponding colored photo" 
    - MLLM可以推断出instruction需要转换成两个prompts，并把任务分类为'sketch coloring'

# 3 Ablation

![image-20241127001314170](imgs/61-Group%20diffusion%20transformers%20are%20unsupervised%20multitask%20learners/image-20241127001314170.png)

## 3.1 Data Scaling

- 缩放数据至 5k, 50k, 500k个groups来研究数据尺度
- 随着数据增加，一致性和prompt follow能力都会提升
- 然而，FID却在逐渐升高，作者认为是因为少量数据容易导致过拟合。

## 3.2 Group Size

- 逐渐把group size的上限增加到2， 4， 8。需要注意，group size翻倍时，self-attn中的sequence length也会double，因此只测试了最多8组。
- 从上表可以看出，更大的group size会显著降低图像质量，一致性和文本跟随能力。
- 原因可能是更难学习，也认为可能是缺乏large group sizes的数据导致的。