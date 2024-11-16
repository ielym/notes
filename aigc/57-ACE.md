# 0 引用

- [ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer](https://arxiv.org/abs/2410.00086)

# 1 介绍

![image-20241116222045565](imgs/57-ACE/image-20241116222045565.png)

- 构建all-in-one的视觉生成模型依赖于（1）多样化的多模态输入格式，（2）多种生成任务。
- 为解决该问题，提出了使用DiT模型 All-round Creator and Editor (ACE) ：
  - 首先分析了大多数视觉生成任务的输入条件，并定义了Condition Unit (CU)。
  - 对于包含多个图像的CUs，引入了一个Image Indicator Embedding来保证图像在指令和在CUs中的顺序。此外，对图像序列使用3D PE，呢个够更好的保证图像的相关性。
  - concat当前CUs的信息和之前生成轮数的信息，构造 Long-context Condtion Unit (LCU) 。通过这种信息链的方式，我们期望模型能够更好的理解用户要求，并生成想要的图像。
- 为了解决缺乏数据的问题，构建了一个数据收集和处理的工作流，共搜集了高质量的结构化CU data 0.7B。
  - 对于需要图像作为条件的任务，主要包含两种方式：
    - 从source image生成：
      - 使用开源的编辑模型进行处理，如风格迁移，增加物体
    - 从大规模数据集中收集成对数据：
      - 包含聚类或分组图像，避免过拟合生成数据分布
  - 对于文本指令，
    - 首先人工构造了不同任务的指令，构造方式为通过模板构造或请求LLM构造。
    - 之后通过训练端到端的指令打标的MLLM来优化指令构造过程，从而能够增加文本指令的多样性。

# 2 方法

![image-20241116222823496](imgs/57-ACE/image-20241116222823496.png)

## 2.1 定义

### 2.1.1 Tasks

把生成和编辑任务分成两大类：

- **Textual modality :** 基于是否直接描述生成图像的内容 或 和输入有差异的文本可以分为：
  - **Generating-based Instructions**
  - **Editing-based Instructions**
- **Visual modality :** 如上图所示，共分成了8个基础的类型。

---

- **Text-guided Generation :** 只需要文本来生成图像，不需要图像条件。
- **Low-level Visual Analysis ：** 从输入图像中提取低级视觉特征，如edge maps，segmentation maps。需要一个源图像和编辑指令。
- **Controllable Generation ：** 是Low-Level Visual Analysis的逆任务。从edge map等生成图像。
- **Semantic Editing :** 根据文本指令，修改输入图像中的一些语义属性，如风格迁移，面部属性等。
- **Element Editing :** 添加，删除和替换特定的物体，并保留其他区域不变。
- **Repainting :** 通过给定的文本指令和mask，擦除并重绘输入图像中的内容。
- **Layer Editing :** 把输入图像分解成不同的layers，每层包含一个主题或背景。或者反过来融合不同的层。
- **Reference Generation :** 根据一个或多个参考图像，分析他们之间相同的元素，并在生成图像中进行生成。

### 2.1.2 Input Paradigm

设计了一个统一的输入范式，定义为 Conditional Unit (CU) ，包含：

- 文本指令 $T$ ，描述生成的要求。
- 视觉信息 $V$ ，$V$ 包含了一系列的图像 $I \in \{ I^1, I^2, ..., I^N \}$ ，同时也可以没有 $I$ （$I = \empty$ ）
- 与视觉信息对应的mask $M \in \{ M^1, M^2, ..., M^N \}$ 。Mask默认为全1的图像

---

因此，CU的形式可以表示为：
$$
CU = \{T, V\} \\
V = \{ [I^1, M^1], [I^2, M^2], ..., [I^N, M^N] \}
$$
$I, M$ 在通道维度concat

---

为了更好的解决复杂的long-context的生成和编辑，历史信息也可以选择性的居合道CU中：
$$
LCU_i = \{ \{ T_{i-m}, T_{i-m+1}, ..., T_i \}, \{ V_{i-m}, V_{i-m+1}, ..., V_i \} \}
$$
$m$ 定义为历史轮数的最大值，$LCU_i$ 表示用于当前第 $i$ 轮生成的 Long-context Condition Unit 。

## 2.2 Architecture

![image-20241116231136615](imgs/57-ACE/image-20241116231136615.png)

如上图(a)所示，ACE基于DiT架构，并插入了三个新的模块：

- Condition Tokenizing
- Image Indicator Embedding
- Long-context Attention Block

### 2.2.1 Condition Tokenizing

- 文本：T5 编码文本指令成1-d的token sequences $y_m$ 
- 图像：VAE编码到latent空间
- mask：一个down-sampling module来把mask缩放到对应的latent的尺寸
- mask和图像在通道维度concat，之后patchified到1-d的visual token序列 $u_{m,n,p}$ 
  - $m$ : 表示CU在LCU中的索引
  - $n$ ：表示每个图像在CU中的索引
  - $p$ ：表示每个token在latent中的空间索引
- 每一个CU内部处理完成之后，我们分别concat所有的visual tokens和所有的textual tokens来构造long-context sequence

### 2.2.2 Image Indicator Embedding

如上图3（b）所示，为了标记文本指令中的图像的顺序，以及区分不同的输入图像，我们编码一些预定义的文本tokens `{image}, {image2}, ..., {imageN}` 到T5的embedding空间，作为 Image Indicator Embeddings (I-Emb) 。

这些标记会被加到对应的图像embedding序列和文本embedding序列：
$$
y'_{mn} = y_m + I-Emb_{m,n} \\
u'_{m,n,p} = u_{m,n,p}
$$
通过这种方式就可以把图像和文本联系起来。

### 2.2.3 Long-context Attention Block

- 对于long-context visual sequence，首先加入 time-step embedding (T-Emb) 
- 之后，使用3D RoPE来区分不同的 spatial- and frame-level的图像embeddings
- 在Long Context Self-Attentionzhong , 单个CU内的所有图像embeddings会做双向attention $u = Attn(u', u')$ 
- 对于cross-attn，只在每个CU内部做cross-attn $\hat{u}_{m,n} = Attn(\mu_{m,n}, y'_{m,n})$

# 3 数据

## 3.1 指令生成

![image-20241117012654747](imgs/57-ACE/image-20241117012654747.png)

需要构造图像编辑的指令，同时使用了 template-based 和 MLLM-based两种方法：

- template-based ：人工构造指令，但多样性差，容易过拟合。
- MLLM-based：可以生成编辑指令，但对于非真实的图像对（不是指图像风格，而是depth等图像）的效果不好

---

因此，需要结合两种方法：

- 对于包含非自然图像的数据，使用template-based方法来生成指令，并使用LLM来进行多次的指令调整，保证生成的指令是unique的。
- 对于包含自然图像的数据，使用MLLM来预测两张图像的差别和共性，然后通过对语义差异和共性的分析，利用LLM生成以语义区别为重点的指令。
- 之后，上述两种方法得到的文本指令和对应的图像会用来训练一个开源的MLLM (InternVL2-26B)，训练数据大约构造了 800000条。
- 之后就可以使用微调后的MLLM模型来处理任意任务的指令生成了。

## 3.2 成对数据构建

![image-20241117012701874](imgs/57-ACE/image-20241117012701874.png)

### 3.2.1 文生图数据

- 数据量约为117M，使用MLLm打标

### 3.2.2 Low-Level Visual Analysis

- 10种任务：segmentation map, depth map, human pose, mosaic image, blurry image, gray image, edge map, doodle image, contour image, and scribble image

![image-20241117013009396](imgs/57-ACE/image-20241117013009396.png)

![image-20241117013017868](imgs/57-ACE/image-20241117013017868.png)

- **Segmentation**： 使用Efficient SAM来标记不同target的mask

- **Depth :** Midas算法来提取深度信息

- **Human-pose** ：RTMPose算法来处理，使用OpenPose’s 17-point进行可视化

- **Mosaic** ：

- **Image Degradation** ：使用Real-ESRGAN的方法进行图像退化，此外还添加了随机噪声

- **Image Grayscale**：opencv进行灰度图变换

- **Edge Detection**：使用opencv的canny

- **Doodle Extraction**：模拟相对粗糙的手绘草图，提取物体的轮廓，忽略其细节。使用PIDNet和SketchNet处理

- **Contour Extraction**：轮廓提取是模拟图像的绘制过程（类似于线稿，但更接近绘画过程种的结构）。使用信息性绘图中的轮廓模块 [Learning to generate line drawings that convey geometry and semantics](https://arxiv.org/abs/2203.12691) 中的contour module进行信息提取

  ![image-20241117013726764](imgs/57-ACE/image-20241117013726764.png)

- Scribble Extraction：包括检索原始线条艺术信息，以捕获图像的草图形式。使用信息性绘图中的轮廓模块 [Learning to generate line drawings that convey geometry and semantics](https://arxiv.org/abs/2203.12691) 中的anime-style module进行信息提取

### 3.2.3 Controllable Generation

- 使用 Low-Level Visual Analysis中的条件

### 3.3.3 Semantic Editing

通过文本修改语义属性，如面部编辑和风格变换

- **面部属性编辑：** 包含面部对齐的和非对其的两类数据

  - **对齐的人脸数据：**

    ![image-20241117014412842](imgs/57-ACE/image-20241117014412842.png)

    - 使用生成模型如InstantID生成像素对齐的人脸数据，并将其与GPT模型相结合以产生各种提示。

    - 随后，我们训练了多个轻量级的二元分类模型，根据图像质量、PPI评分、美学评分和其他指标来清理生成的数据。

    - 之后用ArcFace提取进行相似度计算，保留分数>0.65的
    
    - 最后有一个自迭代训练过程，以生成更高质量的数据。
  
  - **非对齐的人脸数据：**
  
    ![image-20241117014949574](imgs/57-ACE/image-20241117014949574.png)
  
    - 使用人脸检测方法，只保留只包含一个人脸的数据。
  
    - 之后提取面部特征，使用K-means聚了10000类
  
    - 每个类的内部在进行聚类。相似度评分高于0.8和低于0.9的面孔被联系起来，以避免将完全相同的图像分到一组。
  
    - 最后对剩余的数据执行手动注释和重复数据删除
  
- **面部属性迁移：**

  ​	![image-20241117015254193](imgs/57-ACE/image-20241117015254193.png)

  - 微笑、胡须、化妆、染发4个任务。

  - 通过调用阿里云获得了大量相关数据API和训练过的二元分类器为每个类别过滤掉不明显变化的数据。总共获得了1.4M的数据。

### 3.3.4 风格编辑

