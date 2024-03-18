`PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis`

paper: https://arxiv.org/abs/2310.00426



#  1 介绍

SD模型训练慢，本文提出的Pixart-alpha时基于Transformer的 T2I扩散模型：

- 生成质量可以和Imagen, SDXL甚至Midjourney媲美
- 支持最大 $1024 \times 1024$ 分辨率的高分辨率图像生成，但训练成本更低

生成效果如下所示：

![image-20231009133028258](imgs/PixArt-%CE%B1/image-20231009133028258.png)

![image-20231009133037550](imgs/PixArt-%CE%B1/image-20231009133037550.png)

训练成本对比如下：

![image-20231009133116906](imgs/PixArt-%CE%B1/image-20231009133116906.png)

为实现上述目标，文中提出了三个核心思想：

- **Training strategy decomposition** :  把训练任务分解成三个子任务：
  - 学习自然图像的像素分布（使用class-conditon作为条件，而不是文本，使用Imagenet数据集）
  - 学习文本-图像对齐
  - 增强图像质量

- **Efficient T2I Transformer：** 基于 Diffusion Transformer (DiT) [` Scalable diffusion models with transformers` ] ，并合并cross-attention来注入文本条件和class-condition。此外，引入了一种重参数化技术，使得能够用class-condition训练好的模型权重初始化文本条件模型。

- **High-infomative data：** 作者发现LAION数据集的文本标注通常缺乏有效信息，如通常总是描述图像中的部分物体，并且长尾（大量名词的频率都很低），并且作者认为这导致需要数百万此的迭代才能学习到稳定的text-image alignments，这也是为什么SD预训练慢的原因。为解决该问题，作者使用sota的 LLaVA大语言模型来打标，数据集仅使用了SAM，SAM包含丰富多样的各种物体，更适合学习 text-image alignment任务的学习。



# 2 方法

## 2.1 动机

作者认为 T2I 模型训练慢有两个原因导致：训练流程和训练数据。

- 训练流程：为解决训练流程问题，作者把训练过程解耦成三个部分：
  - 学习像素依赖：学习自然图像的像素分布
  - 对齐文本和图像：
  - 高审美质量

- 训练数据：使用一种自动标注流程来产生准确的图像标注，如下图所示：

  ![image-20231009135347088](imgs/PixArt-%CE%B1/image-20231009135347088.png)

## 2.2 训练策略解耦

### 2.2.1 阶段一：学习像素分布

训练class-condition的图像生成模型时比较快速的。此外，发现合适的参数初始化能够显著提升训练效率。因此，我们使用ImageNet数据预训练模型，并作为后续文生图模型的初始化参数。

### 2.2.2 阶段二：文本图像对齐学习

从阶段一的预训练模型到阶段二的文本条件的最主要的挑战就是如何在显著增加的文本概念的情况下，仍任能对齐两个模型。

文中解决该问题的方法时构建非常精确的图像文本对。

### 2.2.3 阶段三：高分辨率高审美图像生成

使用高质量审美数据用于高分辨率图像生成。

## 2.3 高效的 T2I Transfomers

PIXART-alpha使用 Diffusion Transformer (DiT) 作为bakcbone。

![image-20231009143450177](imgs/PixArt-%CE%B1/image-20231009143450177.png)

- **Cross-Attention layer。** 在DiT block中的自注意力机制和feed-forward之间加入多头交叉注意力机制。由于第一个阶段是class-condition,因此没有cross-attention，为了能够有效利用预训练权重，作者把 cross-attention的 proj_out 的权重初始化成0，防止训练早期下一层的输入变化太大。
- **AdaLN-single。** DiT中使用了 AdaLN模块，占了总参数量的 $27\%$ 。因此，本文提出了adaLN-single模块，只使用时间编码作为输入 。具体没有详细看，UNet-based 扩散模型没有这个。
- Re-parameterizaiton。 没详细看，和AdaLN-single耦合了。

## 2.4 数据构建

### 2.4.1 图像-文本自标注

由于LAION质量太差，因此使用LLaVA进行标注。使用的prompt是

`Describe this image and its style in a very detailed manner`

使用SAM数据集，作为第二个阶段的训练数据。

使用JourneyDB数据集和10M网络数据来增强真实风格之外的审美质量。

# 3 实验

## 3.1 训练

- 和 Imagen和DeepFloyd一样，使用T5大语言模型，如4.3B Flan-T5-XXL作为text encoder。
- 使用 DiT-XL/2作为底模
- 把text encoder的文本编码上线从77上调至120，这是由于PIXART-alpha训练用的文本更详细。
- 使用LDM的VAE
- 使用中心裁剪的预处理方式
- 使用SDXL的多尺度训练策略
- AdamW优化器，weight decay = 0.03，固定的2e-5学习率
- 64个V100，大约训了22天

## 3.2 附录训练细节

- 多尺度训练。把图像按照不同高宽比划分成40桶，高宽比从0.25 至 4 。每个batch的图像从一个桶中取出。只在高审美阶段使用多尺度训练。
- 位置编码。由于图像分辨率和高宽比在不断变化，因此在不同的训练阶段使用了DiffFit中的位置编码

- 
