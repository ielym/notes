`Improving Image Generation with Better Captions`

# 1 摘要

证明了在高度详细描述的图像captions上训练，可以显著提升文生图模型的能力。训练了一个定制的图像打标器来重新打标训练数据集。

# 2 介绍

最近基于自回归的生成模型或扩散模型可以把图像生成问题解耦成小的，离散的过程，更易于神经网络学习。

该领域目前最大的挑战是生成模型的可控能力，如单词，单词顺序，或描述的语义。我们把这类问题称为语义跟随（prompt following）。

本文提出了一个新方法来解决语义跟随问题：caption improvement ：

- 首先，学习一个鲁棒的图像打标器，产生详细的，准确的图像描述。
- 然后，用这个打标器打标，产生详细的标注。
- 最后，使用改善的数据集训练文生图模型。

本文主要关注于验证语义跟随的改善，不包含训练和实现细节。

# 3 数据重打标

定义文本图像数据对为 $(t, i)$ ，其中 $i$ 是图像，$t$ 是文本描述。在大规模数据集中，$t$ 通常只关注图像的主体描述，而忽略了背景细节。$t$ 中忽略的信息通常包括：

- 例如厨房里水槽之类的物体的存在，或者人行道上的停车标志，以及对这些物体的描述。
- 物体在场景中的位置，以及这些物体的数量。
- 通用的场景细节，例如场景中物体的颜色和大小。
- 图像中的文本。

更糟糕的是，网络中的描述通常不准确，或描述无关的信息。例如广告或表情包。

## 3.1 构建图像打标器

凸显打标器和传统的预测文本的语言模型非常相似：

- 首先，使用 tokenizer 进行分词，把文本划分成离散的 tokens ，得到词表 $t = [t_1, t_2, ..., t_n]$ 

- 然后，构建一个语言模型来极大化如下的似然函数：
  $$
  L(t) = \sum_j log P(t_j | t_{j-k}, ..., t_{j-1}; \Theta)
  $$
  其中，$\Theta$ 是打标器模型需要优化的参数。

- 为了把语言模型转换成一个打标器，只需要把图像作为条件即可。这个过程的挑战在于图像由数千个像素值组成，使用目前的神经网络效率很低，因此需要压缩表示空间。方便的是，CLIP提供了该过程。一i那次，给定一个预训练好的CLIP Image Embedding 函数  $F(i)$ ，此时目标函数编程了：
  $$
  L(t, i) = \sum_j log P(t_j | t_{j-k}, ..., t_{j-1}; z_j; F(i); \Theta)
  $$

### 3.1.1 Fine-tuning 打标器

- 首先，使用一个小数据集，描述只包含主要物体，生成的描述称为 "schort synthetic captions, SSC" 
- 重复这个过程，创建一个长的，高度详细的描述。不仅包含主要物体，也包含其周围，背景，文本，风格，颜色等信息。生成的描述称为 "descriptive synthetic captions, DSC" 
- 原始标签称为 Alt Text，三种描述的对比如下：

![image-20231022214732736](imgs/36-DALLE3/image-20231022214732736.png)

# 4 验证重描述数据集

主要验证两个问题：

- 使用不同类型的描述对效果的影响
- 最优的合成描述的比例

## 4.1 混合合成和gt描述

似然模型，如文生图扩散模型，容易过拟合数据集的分布。如，一个文生图模型总是在以 a 开头的文本上训练，如果不以 a 开头推理，则不会work的很合适。

在训练打标器时也需要考虑这个问题。我们的captioner模型可能具有许多难以检测的模态行为，但如果在这些标题上进行训练，这些行为将成为我们的文本到图像模型的偏差。例如，在字母大小写中，在标题中出现标点符号的位置(例如，它总是以句号结尾吗?)，标题有多长，或风格倾向，如所有标题都以“a”或“an”开头。

解决该问题最好的方法是让文本更尽皆真实人类可能使用的风格。当使用gt标签时，天然的就是真实人类的风格，因为gt标签就是人写的。因此我们混合合成标签和gt标签。

我们在数据采样时进行混合，随机以一个固定的概率从gt或合成描述中采样。下节将对比不同混合比例对效果的影响。

## 4.2 验证方法

为了验证，我们训练了一个完全相同的 T5-text encoder的图像扩散模型，使用相同的图像。模型训练细节见附录一。所有模型使用 1B 训练数据，bs=2048，训练500000个steps。

训练完成后，我们使用每个模型推理来自验证集的50000个prompt。之后使用CLIP-S结果，计算方式如下：

- 首先，使用开源的 CLIP ViT-B/32图像编码器来产生图像编码 $z_i$ 

- 然后，使用文本编码器产生文本编码 $z_t$ 

- 之后计算CLIP得分：
  $$
  C(z_i, z_t) = 1 - \frac{z_i \cdot z_t}{||z_i|| ||z_t||}
  $$

- 最后5000个结果求平均，并除以100进行缩放。

## 4.3 不同描述类型的结果

我们训练了三个模型：

- 只用 gt 训练一个文生图模型
- 95%的简短的合成描述训练t2i模型
- 95%的复杂合成描述训练t2i模型

我们验证两次模型：

- 一次使用 gt 作为文本来计算 CLIP 得分
- 一次使复杂生成描述作为文本
- 没有使用段的生成描述，因为段的描述和gt很接近。

结果如下：

![image-20231022223605775](imgs/36-DALLE3/image-20231022223605775.png)

- 无论使用哪种描述进行测试，在合成数据集上训练的模型都显著优于在gt上训练的模型。表明使用合成描述没有缺点。

- 使用合成描述训练的模型方差更低。

## 4.4 标签混合比例

为了验证混合比例，我们训了4个模型，合成描述的比例分别为 $65\%, 80\%, 90\%, 95\%$ 。

- 实验进行到一半时，$65\%$ 的效果在所有评价指标上都远远落后于其他比例，因此该实验没有做完就放弃了。 
- 下图表明其他3种比例种，更高的混合数据总能改善CLIP score 。

![image-20231022225811163](imgs/36-DALLE3/image-20231022225811163.png)

## 4.5 高度描述的实际使用

虽然上述实验证明，非常详细的描述进行训练能够提升模型性能，但是，这样训出来的模型也会更适应长的和高度详细的描述分布。

然而，推理时如果不适用长的，高度详细的描述，即使用训练分布之外的描述，结果会比较差。因此，为了最大程度的发挥模型潜力，我们需要从简单描述种采样生成高度详细的描述。幸运的是，大语言模型如  GPT-4 非常擅于这项任务，能够在图像描述中提出合理的细节。

具体怎么利用 GPT-4 生成详细描述的方法见附录 C ，下图展示了生成结果：

![image-20231022230751651](imgs/36-DALLE3/image-20231022230751651.png)

- 使用大语言模型上采样描述不仅可以增加细节，也可以消除复杂的关系（复杂的关系可能对于小的图像生成模型来说难以学习）
- 最终的结果表明，模型通常都会正确的重定向图像。

# 5 DALL-E 3

- 使用 $95\%$ 的合成描述 + $5\%$ 的gt描述
- DALL-E 3 在 DALL-E 2的基础上又很多改进，一些改进没有在论文种介绍，并且优于资源限制，没有做消融实验。

# 6 不足

## 6.1 空间感知

虽然DALL-E 3在提示跟随方面迈出了重要一步，但它在对象放置和空间感知方面仍有困难。例如，使用“在左边”、“在下面”、“在后面”等词是相当不可靠的。这是因为我们的合成captioner也有一个缺点:它在表示对象位置方面不可靠，这反映在我们的下游模型中。

## 6.2 文字生成重定向

在构建标题器时，我们特别注意确保它能够在生成的标题中包含图像中发现的突出单词。因此，DALL-E 3可以在提示时生成文本。在测试过程中，我们注意到这种能力是不可靠的，因为单词有缺失或额外的字符。

我们怀疑这可能与我们使用的T5文本编码器有关:当模型在提示中遇到文本时，它实际上看到了表示整个单词的标记，并必须将这些标记映射到图像中的字母。在未来的工作中，我们希望探索字符级语言模型的条件作用，以帮助改善这种行为。

## 6.3 一致性

我们观察到，我们的合成标题容易使图像的重要细节产生幻觉。例如，给定一幅花的植物图，打标器通常会产生幻觉，并将其放入标题中，即使这些细节以文本形式出现在图像中。我们在描述鸟类的图片时也观察到类似的行为:有的物种会出现幻觉，有的则根本不被提及。

这对我们的文本到图像模型产生了下游影响:DALL-E 3在为上述特定术语生成图像方面不可靠。我们相信对标题器的进一步改进应该能够进一步改进我们的文本到图像模型。

# 7 附录

## 7.1 A Image Decoder

- 图像解码器是一个文本作为条件的U-Net LDM，共有3个阶段
- 8x下采样的VAE
- timestep condition使用 modulated GN，科学系的缩放和偏置分别应用到GN的输出层。
- text condition 使用 T5 XXL文本编码器。

## 7.2 DALL-E 3 latent decoder

- 在VAE的隐空间上训练diffusion decoder
- 使用diffusion decoder能够改善细节，如文字或人脸。

- 训练好后，使用一致性蒸馏，把去噪步数降低到两步。

## 7.3 使用GPU-4描述上采样

```
You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example , outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning , as described. You will be prompted by people looking to create detailed , amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
```

```
There are a few rules to follow:
- You will only ever output a single image description per user request.
- Sometimes the user will request that you modify previous captions. In this case , you should refer to your previous conversations with the user and make the modifications requested.
- When modifications are requested , you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
- Other times the user will not want modifications , but instead want a new image. In this case , you should ignore your previous conversation with the user.
- Image descriptions must be between 15-80 words. Extra words will be ignored.
```

```
{"role": "user", "content ": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: "a man holding a sword \""},
```

```
{"role": "assistant", "content ": "a pale figure with long white hair stands in the center of a dark forest , holding a sword high above his head. the blade glows with a blue light , casting a soft glow on the trees and bushes surrounding him."},
{"role": "user", "content ": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: "make the light red"},
{"role": "assistant", "content ": "a pale figure with long white hair stands in the center of a dark forest , holding a sword high above his head.
the blade glows with a red light , casting a warm glow on the trees and bushes surrounding him."},
{"role": "user", "content ": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: "draw a frog playing dominoes"},
{"role": "assistant", "content ": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth , and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest , with a large tree in the background ."}
```

