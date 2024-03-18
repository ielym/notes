# 1 贡献

- 提出了InstructPix2Pix，通过指令来修改图像
- 由于训练数据获取比较困难，提出了一种结合大语言模型 (GPT-3) 和 文生图模型 (SD) 来生成成对数据的方法。
- 使用生成的成对数据，训练了一个条件扩散模型，能够给定一张图像和文本指令来进行图像编辑。

# 2 Method

![image-20230913142907434](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913142907434.png)

训练包含两步：

- 根据文本指令，生成编辑前-编辑后 的成对训练数据
- 使用生成的数据，训练一个图像编辑的扩散模型

尽管模型是在生成的数据上训练的，但是在真实图像上仍然具有较好的泛化性。

## 2.1 生成多模训练数据集

- 微调GPT-3，输入一个描述源图像的prompt，输出一个编辑指令以及一个描述编辑后的图像的prompt
- 使用一个文生图模型，根据原图像的prompt和编辑后图像的prompt，生成对应的两张图像

### 2.1.1 生成指令和成对的文本描述

只在文本领域进行，不涉及图像。

![image-20230913155559204](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913155559204.png)

如上图所示，给微调的GPT-3一个输入caption，GPT-3会输出一个指令，以及该指令修改后图像的prompt。

微调GPT-3的数据来自于小型的人类手写的三元组数据：输入描述，编辑指令，输出描述：

- 从LAION-Aesthetics V2 6.5+数据集中挑选了700张图像的描述作为输入描述
- 手写指令和输出描述

![image-20230913160043400](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913160043400.png)

微调了GPT-3一个epoch。

### 2.1.2 从成对的文本描述中生成成对的图像

由于即使极小的文本描述修改，生成图像的差别也可能非常大，不适用于图像编辑任务。因此，本文使用了P2P来生成成对图像：

![image-20230913161425904](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913161425904.png)

然而，比如把小自行车替换成大卡车这种编辑，涉及到修改的区域过大，可能直接使用P2P的效果也不好。但是P2P可以控制编辑时的步长（在Word Swap）中，如下图所示：

![image-20230913162109854](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913162109854.png)

其中，$\tau$ 对应本文的 $p$ 。

因此，本文对于每个成对的prompt，都生成了100张图像，每张图像随机从 $p \sim U(0.1, 0.9)$ 采样，并使用CLIP-based metric [Stylegan-nada: Clip-
guided domain adaptation of image generators] 进行过滤，该metric计算了两张图像变化的一致性。

### 2.1.3 InstructPix2Pix

- 有两个条件：
  - 文本条件：指令编码 $C_T$
  - 图像条件：原始图像经过VAE编码，$E (C_I)$

- 文本条件输入方式与SD相同，图像条件输入方式是与latent按通道拼接（Unet输入8通道）

- 损失如下：

  ![image-20230913164015893](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913164015893.png)

- 使用SD的权重初始化，新加的4个图像条件的通道初始化成0.

### 2.1.4 CFG

#### 2.1.4.1 训练

- 文本条件 5% 置空
- 图像条件 5% 置空 (隐编码置0)
- 文本条件+图像条件 5% 置空

![image-20230913164823661](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913164823661.png)

#### 2.1.4.2 推理

![image-20230913164301492](imgs/31-InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20Instruction/image-20230913164301492.png)
