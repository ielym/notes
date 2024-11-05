# PixWizard: Versatile Image-to-Image Visual Assistant with Open-Language Instructions

# 0 引用

- 论文：[PixWizard: Versatile Image-to-Image Visual Assistant with Open-Language Instructions](https://arxiv.org/abs/2409.15278)

# 1 介绍

- 通过语言指令进行图像生成，编辑和图像迁移的方法。
- 使用统一的框架进行多种视觉任务。e.g., t2i, image restoration, image grounding, dense image prediction, image editing, controllable generation, inpainting/outpainting, etc. 
- 构建Omni Pixel-to-Pixel Instruction-Tuning Dataset

---

目前general visual assistant又两个主要的方向：

- **diffusion-based :** 
  - 利用t2i模型作为基础模型来处理不同视觉感知任务，如InstructP2P, InstructDiffusion, InstructCV等。
  - 缺点：
    - 能力受限
    - 性能低于task-specific模型
- **in-context learning :** 
  - 关注visual prompting，使用Prompting方法和in-context的方式生成visual outputs。e.g., Painter, PromptDiffusion, LVM
  - 缺点：
    - 指令跟随能力较差

---

**PixWizard :** 

- **Task Unification :** 视觉任务的种类较多，包括像素级的，定位，分类，分割等。很难进行统一的表示。本文统一转换成视觉任务。
- **Data Construction :** 构建了一个30M的数据集，支持5个主要的能力：
  - **image generation**，包含 text-to-image, controllable, inpainting, outpainting
  - **Image editing**
  - **Image restoration**, 包含deraining, desnowing, deblurring, super-resolution
  - **Image grounding**, 根据用户指令定位物体
  - **Dense image prediction**, 包含depth estimation, surface normal estimation, pose estimation, semantic segmentation, image-to-canny/HED/sketch/Line-Art conversions
- **Architecture Design :** flow-based DiT作为base model。



# 2 Dataset

**Omni Pixel-to-Pixel Instruction-Tuning Dataset**

- 30M图像

![image-20241104231521834](imgs/PixWizard/image-20241104231521834.png)

---

- **Image Restoration.**  

  ![image-20241104234004881](imgs/PixWizard/image-20241104234004881.png)

  - including (1) Denoising, (2) Deraining, (3) Demoireing, (4) Dehazing, (5) Deblurring, (6) Desnowing, (7) Deshadowing, (8) Low-Light Enhancement, (9) Face Restoration, (10) Watermark Removal, and (11) Super Resolution

- **Image Grounding**

  - 根据text promtp定位或高亮物体的特殊区域 :
    - Segmentation Referring
      - 指令格式：`Please mark the pixels in {color} based on the referring description: {caption}`
      - label的mask设置50%的不透明度
    - Box Detection Referring
      - 指令格式：`Mark the specified area with a bounding box in {color}: {caption}.`
    - Binary Mask Prediction
      - 指令格式：`"Generate a binary mask for the described object: {caption}` 

- **Controllable Generation**

  - Canny/HED/Depth/Sketch/Pose/Semantic Segmentation/Normal Map/Line-art   to image
  - 每个类型都构造了约1M的数据对

- **Dense Image Prediction**

  - depth estimation, surface normal estimation, and semantic segmentation等任务可以直接使用RGB images作为label
  - human pose maps, sketches, HED boundaries, canny edge maps, and cartoon line art使用可视化图像作为label

- **Image Editing**

  ![image-20241104234506905](imgs/PixWizard/image-20241104234506905.png)

  - 使用开源数据集

- **Inpainting**

  - 使用随机black or white masks。包含circles, rectangles, and free-form patterns的mask

- **Outpainting**

  - 随机crop图像的中心区域，周围区域用black or white的mask。crop区域使用不同高宽比的矩形。

- **Text-to-Image Generation**

  - 从互联网上搜集的高质量数据，使用MLLMs打标
  - 为了区分t2i任务，和文本共同输入的还有一个white or black的空白图像作为标记

- **Open-language Instruction**

  - 上述每个任务人工写6-10个prompt，然后用GPT-4o生成大量的变体。

  ![image-20241104235840797](imgs/PixWizard/image-20241104235840797.png)

![image-20241104235859409](imgs/PixWizard/image-20241104235859409.png)

![image-20241104235912783](imgs/PixWizard/image-20241104235912783.png)

# 3 PixWizard

- finetune Lumina-Next-T2I

## 3.1 Text Encoder

- 使用Gemma-2B来encode text prompt

- 由于存在多个任务，只依赖text embedding不足以让模型准确的区分不同任务，因此还额外加了一个CLIP text encoder：

  - 对CLIP的text embedding做global average pooling，得到粗粒度的task embedding
  - 把task embedding加到timestep embedding上
  - 如下图所示，用这种方法得到的task embedding更能区分不同任务

  ![image-20241105000459802](imgs/PixWizard/image-20241105000459802.png)

## 3.2 Structural-Aware Guidance

- 为了更好的保持结构，使用SDXL的VAE输入image latent，并和IP2P一样，拼接到channel维度，新增的权重初始化成0

## 3.3 Semantic-Aware Guidance

- 使用CLIP L/14-336提取semantic image embedding。

- 在PixWizard block中使用cross-attn进行注入，并且假如了rope和门控机制：

  ![image-20241105001742202](imgs/PixWizard/image-20241105001742202.png)

  - $Q_i, K_i, V_i$ 是self-attn，$Q_t, K_t, V_t$ 是文本，$Q_{ci}, K_{ci}, V_{ci}$ 是clip embedding
  - $\hat{Q_i}, \hat{K_i}$ 表示RoPE
  - 门控输出初始化成0

- 为了解决计算量过大的问题，采用了 Task-Aware Dynamic Sampler，选择语义最相关的token（包含4层线性层和激活函数，做rank）。方法来自于 DynamicViT。

  ![image-20241105002204476](imgs/PixWizard/image-20241105002204476.png)

## 3.4 Any Resolution

- 训练时基于[512^2, 768^2, 1024^2] 打包bucket数据
- 推理时使用NTK-Aware Scaled RoPE 和 sandwich normalization

## 3.5 训练策略

- Stage1：
  - 使用预训练权重，新增的层随机初始化。
  - 从数据量较少的任务开始训练，给不同数据量的任务一个sampling weight，每个任务都大概训20k个样本
  - 之后，随机的从其他任务中挑选样本假如寻来你
  - 一共训4 epochs
- Stage2：
  - 使用所有数据进行训练，为了平衡不同任务，手动给不同任务指定采样权重
  - t2i任务和其他任务的比例是 1:1 
  - 训练约20M个样本

# 4 实验

![image-20241105003036394](imgs/PixWizard/image-20241105003036394.png)

![image-20241105003042487](imgs/PixWizard/image-20241105003042487.png)

![image-20241105003049465](imgs/PixWizard/image-20241105003049465.png)

- 更多结果见论文

## 4.1 Ablation Study

![image-20241105003230165](imgs/PixWizard/image-20241105003230165.png)

### Structural-Aware vs. Semantic-Aware Guidance

- M1：只用structure-aware module
- M2：只用semantic-aware moudle

### Influence of Dynamic Semantic Tokens Sampling

- 使用动态tokens (DSTS) 影响不大，因此不需要使用全部token来增加计算量。

### Influence of Two-Stage Training and Data Balancing

- 有效改善了小数据集上的效果

# 5 Discuss

- 不支持多张图像输入
- 分割和检测任务还有很大的提升空间
- 可以使用更好的text encoder实现精确的指令跟随
- 使用更好的t2i模型，如SD3, FLUX