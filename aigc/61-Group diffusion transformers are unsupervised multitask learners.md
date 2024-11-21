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

 