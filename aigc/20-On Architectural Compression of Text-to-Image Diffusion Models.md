`On Architectural Compression of Text-to-Image Diffusion Models`

![image-20230726000957185](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230726000957185.png)

# 1 Motivation

为了解决SD模型计算量大的问题，最近的方法通常通过减少采样步数，或量化的方式。不同于这些方法，本文采用了经典的网络压缩，通过引入 block-removed knowledge-distilled Stable Diffusion Models (BK-SDMs) ，剔除了部分残差和注意力模块，减少了30%的参数量,MAC和延时。

我们只使用了 0.22M 的LAION数据集（不到总量的0.1%）进行蒸馏预训练，只使用单卡A100。尽管训练资源有限，但我们的效果和原始的SDM非常接近。

此外，我们证明了我们的轻量级模型能够使用DB很好的微调。

通用生成能力和个性化DB微调的效果如下图所示：

![image-20230725230844238](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230725230844238.png)

SD-v1中的主要MAC和参数量分布：

![image-20230725231028553](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230725231028553.png)

# 2 BK-SDM

由于U-Net是SDM中计算量最大的部分，因此我们压缩UNet。UNet包含多个去噪时间步，我们关注的是降低每个时间步的计算量。

## 2.1 压缩UNet网络结构

![image-20230725231314515](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230725231314515.png)

如上图所示，本文共压缩了3种不同大小的网络，参数量如下所示：

- BK-SDM-Base ：0.76B。使用了更少的blocks。
- BK-SDM-Small : 0.66B。使用了更少的blocks，移除整个mid-stage
- BK-SDM-Tiny ：0.50B。使用了最少的blocks,移除整个mid-stage

### 2.1.1 Fewer blocks in the down and up stages

该方法基本和DistilBERT对齐了，减半了网络层的数量来改进计算效率，并使用原始的权重来初始化压缩后的模型。

在原始UNet种，每个stage都使用相同的空间分辨率，包含多个blocks，大多数stage都跑含一组 R+A (Residual + Cross-Attention) blocks。我们假设去除一些不需要的R+A，并使用如下策略：

- 对于下采样阶段，我们保留第一个R-A对，并移除第二组R+A。这是由于第一组R+A需要处理上一层进行的空间变换后的特征，因此作者认为该组R+A比第二组更重要。这种方法不损害原始UNet的维度，并且能够使用原始的预训练权重。
- 对于上采样阶段，我们保留第三组R+A对。这使得我们能够使用多个下采样Stage的输出特征，以及对应的上采样-下采样的skip connection。此外，和下采样类似，还保留了第一组R+A来处理变化的空间维度特征。即，上采样阶段只移除第2个R+A对。

### 2.1.2 Removal of the entire mid-stage

作者发现移除掉原始UNet的整个mid-stage，并不会造成明显的生成质量的退化，并且十分有效的减少了参数量（ResNet的通道维度太大）。

如下图所示，参数量减少了11.3%，但即使不重新训练，也不微调，FID并没有上升太多。

![image-20230725232926878](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230725232926878.png)

![image-20230725233035058](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230725233035058.png)

此外，如上图所示，即使不重新训练，移除掉整个mid-stage也不会太影响生成效果。

### 2.1.3 Further Removal of the Innermost Stages

为了更进一步的压缩，innermost的down和up stages可以从BK-SDM-Small中进一步的移除，最终获得 BK-SDM-Tiny模型。这意味着模型使用了更大的空间维度，其skip-connection的角色更加重要。

如上图网络结构所示：

- 倒数第二个阶段的下采样层被移除了
- 倒数第一个阶段的下采样层和第一个阶段的上采样层都被移除了
- mid-stage被整体移除了。



## 2.2 基于蒸馏的预训练

为了通用意图的T2I，我们训练了压缩Unet，来模拟原始Unet的表现。输入仍然使用pretrained-and-frozen的编码器来获得UNet的输入：

### 2.2.1 去噪损失

给定隐编码 $z$ 以及对应的文本编码 $y$ ，该loss和原始UNet训练一致：
$$
L_{Task} = \mathbb{E}_{z, \epsilon, y, y} [ || \epsilon - \epsilon_S(z_t, y, t) ||_2^2 ]
$$
其中，$\epsilon$ 是噪声真值，$\epsilon_S$ 是UNet的预测噪声。

### 2.2.2 输出值蒸馏

对于教师模型和学生模型的输出预测噪声进行蒸馏：
$$
L_{OutKD} = \mathbb{E} [ || \epsilon_T(z_t, y, t) - \epsilon(z_t, y, t) ||_2^2 ]
$$

 ### 2.2.3 特征蒸馏

$$
L_{FeatKD} = \mathbb{E} [ \sum_l || f_T^l(z_t, y, t) -  f_S^l(z_t, y, t)||_2^2 ] 
$$

### 2.2.4 损失汇总

$$
L = L_{Task} + \lambda_{OutKD} L_{OutKD} + \lambda_{FeatKD} L_{FeatKD}
$$

# 3 实验

## 3.1 数据集和验证指标

### 3.1.1 预训练

使用了0.22M的LAION-Aesthetics V2.6.5+ 的图像-文本对来训练压缩后的SDM。远低于原始SDM的训练数据。

预训练：50K-iteration

DB:PEFT仓库的代码，训练 800 个iters。

### 3.1.2 消融实验

![image-20230726001044843](imgs/20-On%20Architectural%20Compression%20of%20Text-to-Image%20Diffusion%20Models/image-20230726001044843.png)

- 对比N1和N2：使用预训练权重初始化相对于随机初始化，其他配置都相同，但结果有显著的差异。
- 对比N2，N3，N4：蒸馏输出和特征的效果最好。
- 对比N4, N5 ：增大BS训练，IS和CLIP score更好，但是FID稍有下降。作者认为更大的bs能够更好的增强模型的理解能力。
