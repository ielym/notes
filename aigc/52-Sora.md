# Video generation models as world simulators

# 0 引用

- project page: https://openai.com/research/video-generation-models-as-world-simulators

Scalable diffusion models with transformers

Scaling autoregressive models for content-rich text-to-image generation

Vivit: A video vision transformer

An image is worth 16x16 words: Transformers for image recognition at scale

Videogpt: Video generation using vq-vae and transformers

# 1 介绍

OpenAI 发布了Sora文生视频/图像模型，探索了在large-scale的视频数据上训练生成模型。Sora是一个基于文本条件的扩散模型，基于不同的时间维度，既可以实现视频生成，也可以实现图像生成（time duration = 1） 。此外，Sora还支持不同分辨率和高宽比的视频生成。

Sora基于spacetime patches的Transformer架构，并在隐空间进行编码。实验发现scale-up 视频生成模型是一个非常有潜力的方向来模拟物理世界。

技术报告主要关注两个方向：

- 把所有类型的视觉数据转换成一种统一的表示形式，从而支持生成模型的大尺度训练。
- Sora的能力和限制的质量评估。
- 不包含模型和实现细节。

最近的一些其他工作，如循环网络，GANs，自回归transformer和扩散模型等都只关注视觉数据的一个很小的类别，如只关注短视频，或只能生成固定尺寸的视频。Sora可以生成分钟级别的视频，并且支持不同时长，高宽比和分辨率。

# 2 方法

## 2.1 Turning visual data into patches

受LLM的启发，LLM能够把不同模态的文本，数学公式，代码，和不同类型的自然语言都编码成tokens，从而能够充分利用互联网规模的数据进行训练。因此我们考虑如何让生成模型在视觉数据上也能够因此受益。类似于text tokens，Sora利用 visual pathes。我们发现patches有更高的可缩放能力和更高效的表达能力来在不同类型的视频和图像数据上训练生成模型。

我们首先通过把视频压缩到更低维的隐空间，之后分解到时空patches [At a high level, we turn videos into patches by first compressing videos into a lower-dimensional latent space,[19](https://openai.com/research/video-generation-models-as-world-simulators#fn-19) and subsequently decomposing the representation into spacetime patches.]：

![image-20240217002539164](imgs/52-Sora/image-20240217002539164.png)

## 2.2 Video compression network

训练了一个网络来减少视觉数据的维度。该网络使用原始视频作为输入，并输出lantent representation来同时压缩空间维度和时间维度。Sora是在该压缩的latent space上进行训练和生成的。同时也训练了一个对应的decoder用于把latents映射回像素空间。

## 2.3 Spacetime Latent Patches

对于一个压缩后的视频输入，我们提取一系列的时空pathces作为transformer的tokens。这个策略也能偶用于图像，因为图像可以看作是一个单帧的视频。

patch-based表示方法能够使得Sora在视频和图像上进行训练，也可以使用不同分辨率进行训练，并且支持不同时长和高宽比。

推理时，可以通过排列不同随机初始化的patches来控制生成尺寸。

## 2.4 Scaling transformers for video generation

Sora是一个扩散模型，输入噪声pathces，并使用condition信息（如文本描述）作为控制，来训练预测原始clean patches的能力。

![image-20240217003836264](imgs/52-Sora/image-20240217003836264.png)

Sora是一个 diffusion transformer （DiT），Transformer被证明具有非凡的缩放属性across a variety of domains。

我们发现DiT也可以有效地缩放为视频模型。下面，我们展示了随着训练的进行，具有固定种子和输入的视频样本的比较。随着训练计算量的增加，样本质量显著提高。

![image-20240217003615168](imgs/52-Sora/image-20240217003615168.png)

## 2.2 Variable durations, resolutions, aspect ratios

过去图像和视频生成的方法通常把数据通过resize, crop或抽帧得到标准的尺寸，如256x256的4s的视频。但是我们发现在原始尺寸上训练有几个优点。

### 2.2.1 Sampling flexibility

Sora可以采样 1920x1080, 1080x1920和二者之间的任意尺寸的视频。有两个优点：

- 可以直接基于不同设备进行不同高宽比的生成。
- 多尺度的方式也可以在生成full resolution之前首先用较小的分辨率进行快速的内容生成。

![image-20240217003747984](imgs/52-Sora/image-20240217003747984.png)

### 2.2.2 Improved framing and composition

实验发现，使用原始高宽比训练的方法具有更好的构图和整体结构。作为对比，我们还训练了一个square-crop的模型，对比如下：

![image-20240216224213242](imgs/52-Sora/image-20240216224213242.png)

## 2.3 Language understanding

训练t2v生成模型需要大量的视频数据以及对应的文本描述。我们使用DALLE-3相同的re-caption技术：

- 首先训练一个高度详细的打标器模型来对视频生成文本描述。实验发现详细视频描述改善了文本跟随能力，同时也改善了视频的整体生成质量。
- 类似于DALLE-3，我们也使用了GPT来把用户短的描述扩展成更长的详细描述。这使得Sora能够生成准确跟随用户prompt的高质量视频。

## 2.4 Prompting with images and videos

Sora除了支持t2v之外，还支持已经存在的图像和视频作为输入。这个能力能够保证Sora进行广泛的图像和视频编辑任务：创建完美的循环视频，动画静态图像，扩展视频向前或向后的时间等。

### 2.4.1 Animating DALLE images

基于图像生成视频的示例：

![image-20240216225045204](imgs/52-Sora/image-20240216225045204.png)

![image-20240216225107316](imgs/52-Sora/image-20240216225107316.png)

![image-20240216225244601](imgs/52-Sora/image-20240216225244601.png)

### 2.4.2 Extending generated videos

Sora也能扩展视频，可以在当前视频之前进行扩展，也可以在当前视频之后进行扩展。

![image-20240216225907116](imgs/52-Sora/image-20240216225907116.png)

### 2.4.3 Video-to-video editing

扩散模型具有大量的方法来使用文本描述来编辑图像和视频。下面是利用SDEdit来让Sora进行视频编辑的方法，这个方法能够使Sora基于zero-shot来转换风格和环境：

![image-20240216230226724](imgs/52-Sora/image-20240216230226724.png)

### 2.4.4 Connecting videos

我们还可以使用Sora在两个输入视频之间逐渐插入，在具有完全不同主题和场景构图的视频之间创建无缝过渡。在下面的例子中，中间的视频在左边和右边对应的视频之间插入：

![image-20240216230459599](imgs/52-Sora/image-20240216230459599.png)

## 2.5 Image generation capabilities

Sora也能够用来生成图像，通过把temporal extent 设置成1来实现。图像生成可以支持最大2048x2048的任意尺寸：

![image-20240216230826998](imgs/52-Sora/image-20240216230826998.png)

## 2.6 Emerging simulation capabilities

我们发现视频模型在大规模训练时表现出许多有趣的突发能力。这些能力使得Sora能够仿真物理世界中的人，动物和环境的一些方面。这些属性的出现没有任何明确的3D、物体等的归纳偏差——它们纯粹是尺度现象。

### 2.6.1 3D consistency

3D可以生成动态视角移动的视频，包括移动和旋转，人物和场景中的元素在3D空间移动时能够保持一致性：

![image-20240216232132209](imgs/52-Sora/image-20240216232132209.png)

### 2.6.2 Long-range coherence and object permanence

视频生成的一个主要挑战就是长视频生成时的时序一致性。我们发现Sora通常（不是总是）能够有效的建模长-短期依赖。比如，即使当人物，动物或其他物体移出了视角之外，后面也依然能够保持一致性。同样，它可以在单个样本中生成同一角色的多个镜头，在整个视频中保持其外观。

![image-20240216232537788](imgs/52-Sora/image-20240216232537788.png)

![image-20240216232543636](imgs/52-Sora/image-20240216232543636.png)

![image-20240216232559082](imgs/52-Sora/image-20240216232559082.png)

### 2.6.3 Interacting with the world

Sora有时可以建模物体之间的相互作用。比如，画笔在画布上的笔触，并能长时间保持下去，或者吃汉堡的交互：

![image-20240216232800115](imgs/52-Sora/image-20240216232800115.png)

![image-20240216232814177](imgs/52-Sora/image-20240216232814177.png)

### 2.6.4 Simulating digital worlds

Sora还能够模拟人工过程，比如视频游戏。Sora可以在高保真度渲染世界及其动态的同时，用基本策略控制《我的世界》中的玩家。这些功能可以通过向Sora提示“我的世界”的描述而获得。

![image-20240216232937843](imgs/52-Sora/image-20240216232937843.png)

这些功能表明，视频模型的持续缩放是发展物理和数字世界以及生活在其中的物体、动物和人的高性能模拟器的一条有希望的道路。

# 3 讨论

Sora也存在一些局限性：

- 不能准确的建模一些基本的物理交互，如玻璃破碎，吃食物，
- 有时不能正确生成物体状态的变化。

我们相信，Sora今天所拥有的能力表明，视频模型的scaling是一条很有前途的道路，可以开发出物理和数字世界的模拟器，以及生活在其中的物体、动物和人。