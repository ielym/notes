`An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion`

code : https://textual-inversion.github.io

# 1 介绍

Text Inversion可以使用3 - 5 张用户提供的概念，比如物体或风格，来学习一个新的单词来表示该概念，具体是用一个text embedding来表示的。实验发现，一个单个的embedding就可以表示不同的概念。

优点：使用db等训练特定概念时，容易遗忘之前学习的内容，而 text inversion 只加入一个额外的 text embedding，不会改变其他任何地方。

# 2 方法

![image-20230925173605090](imgs/33-Text_Inversion/image-20230925173605090.png)

- 定义一个 pseudo-word $S_*$ ，也称为 placeholder string，来表示新概念。需要注意新概念是词表中不存在的新单词。
- 最终的目的是学习到新单词 $S_*$ 所对应的 text embedding $v_*$ 

为了学习到 $v_*$ ，我们使用少量图像（通常3-5张）。我们通过直接优化 LDM的损失来作为 $v_*$  学习的损失函数。

为了控制生成，收CLIP ImageNet启发，我们随机采样中性的上下文文本，这些上下文如 `A photo of S*` ，`A rendition of S*` 等等。论文附录中提供了模板：

```
• “a photo of a S∗.”,
• “a rendering of a S∗.”,
• “a cropped photo of the S∗.”,
• “the photo of a S∗.”,
• “a photo of a clean S∗.”,
• “a photo of a dirty S∗.”,
• “a dark photo of the S∗.”,
• “a photo of my S∗.”,
• “a photo of the cool S∗.”,
• “a close-up photo of a S∗.”,
• “a bright photo of the S∗.”,
• “a cropped photo of a S∗.”,
• “a photo of the S∗.”,
• “a good photo of the S∗.”,
• “a photo of one S∗.”,
• “a close-up photo of the S∗.”,
• “a rendition of the S∗.”,
• “a photo of the clean S∗.”,
• “a rendition of a S∗.”,
• “a photo of a nice S∗.”,
• “a good photo of a S∗.”,
• “a photo of the nice S∗.”,
• “a photo of the small S∗.”,
• “a photo of the weird S∗.”,
• “a photo of the large S∗.”,
• “a photo of a cool S∗.”,
• “a photo of a small S∗.”,
```

此外，Diffusers中提供了一些风格的模板：

```
        self.templates_style = [
            "a painting in the style of {}",
            "a rendering in the style of {}",
            "a cropped painting in the style of {}",
            "the painting in the style of {}",
            "a clean painting in the style of {}",
            "a dirty painting in the style of {}",
            "a dark painting in the style of {}",
            "a picture in the style of {}",
            "a cool painting in the style of {}",
            "a close-up painting in the style of {}",
            "a bright painting in the style of {}",
            "a cropped painting in the style of {}",
            "a good painting in the style of {}",
            "a close-up painting in the style of {}",
            "a rendition in the style of {}",
            "a nice painting in the style of {}",
            "a small painting in the style of {}",
            "a weird painting in the style of {}",
            "a large painting in the style of {}",
        ]
```

按照 Diffusers中的流程，Text Inversion的流程如下：

- 定义一个tokenizer中不存在的新单词的字符串 $S_*$ 
- 初始化 $S_*$ 的 tokenizer id (通常是现有词表按顺序向后+1) 和 embedding （可以随机初始化，也可以使用现有词表中一个类似的embedding进行初始化）
-  训练时，新概念的promot 构成为 `template + S_*` ，如  `a close-up photo of a S∗.`
- 之后正常编码，训练，算loss
- 训练每次迭代时，只修改 $S_*$ 的 embedding，其他 token 的不变。

# 3 实验

- 使用现有的单个单词的embedding初始化新单词的embedding，现有单词需要时新单词的粗略描述，如雕像，猫等
- 使用 2 x V100，bs=4
- 基础学习率是 5e-3，并根据卡数和bs数缩放学习率，实际使用的学习率是 0.005 x 2 x 4 = 0.04 
- 训练 5000 个step 。此外，作者发现对于一些概念更高的学习率+更少的步长可以获得更好的效果。