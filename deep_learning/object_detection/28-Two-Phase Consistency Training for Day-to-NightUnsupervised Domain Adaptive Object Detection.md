

# 2 方法

定义：

- $D_s = \{I_s, C_s, B_s\}$ 表示白天数据，$s$ 是图像索引，$I, C, B$ 分别表示图像，类别和边框
- $D_t = \{I_t\}$ 表示夜晚数据，只有图像没有标签，$t$ 是索引

![image-20230418194206258](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418194206258.png)

 网络结构如上图所示：

- Student同时使用白天和夜晚的数据训练，teacher网络关注夜晚图像来产生伪标签。

- teacher网络是student网络的EMA，平滑系数为 $0.9996$

- 刚开始没有教师网络，学生网络先自己预训练几个epoch，预训练过程中，使用的是白天数据增强策略。这个阶段的Loss就只有学生网络自己的 two-stage 检测的loss ：$L_{sup} = L_{rpn}(B_s, I_s) + L_{roi} (B_s, C_s, I_s)$ ，RPN计算objectness + 边界框回归的loss，RPN计算分类+边框回归loss。

- 当预训练完成之后，使用学生网络的伪标签来产生教师网络。

- 之后的训练包含两个阶段：

  - 阶段一：

    - 教师网络输入夜晚图像，产生伪标签，并通过设定的置信度阈值 (0.8) 进行过滤。这是为了保证给到student网络的都是高质量的伪标签。
    - 教师网络产生的伪标签，与学生网络 RPN产生的region proposals合并到一起，之后送到学生网络的ROI网络中进行预测

  - 阶段二：

    - 教师网络使用相同的合并后的 region proposals来产生一些伪标签的匹配对。这里匹配对的意思是：对于这些合并的 region proposals，学生网络会送到学生网络的的ROI网络，教师网络也会送到教师网络的ROI网络，同一个 region proposal在两个网络中最终的预测结果组成一组匹配对 。

    - 这些匹配对需要计算无监督损失：
      $$
      L_{unsup} = L_{rpn}^{obj}(C_p^*; I_t) + L_{cons}(C_p^*; I_t)
      $$
      其中，$L_{rpn}^{obj}$ 是RPN的objectness损失，$L_{cons}$ 是加权的KL散度，后面会解释。

## 2.1 两阶段一致性

- 由于教师模型是学生模型的EMA，而学生模型只在白天数据上训练过，因此教师模型对夜晚图像产生的伪标签的质量也不会很高。如果使用hard threshold来过滤教师模型产生的伪标签，会导致保留下来的都是简单样本，无法使学生模型学习到 harder areas (e.g. darker)。

- 此外，教师模型由于缺乏 hard samples，比如夜晚分布的先验知识，因此可能产生高置信度，但是却错误的伪标签。这些伪标签又会给到学生网络，学生网络的EMA又给教师网络，这会导致不断恶性循环。本文实验中，发现这些高置信度但错误的伪标签都发生于 dark/glare的区域。

为了解决上述问题，本文提出了一种两阶段的方法：

- Phase 1 :
  - 没有标签的夜晚图像 $I_t$ 作为教师网络的输入，使用教师网络的RPN产生伪标签。之后使用置信度阈值来过滤低质量标签，仅保留高质量伪标签 $(C_p, B_p)$ 
  - 之后，同一张 没有标签的夜晚图像 $I_t$ 作为学生网络的输入，RPN会产生一系列预测框 $RPN_{student}(I_t)$ ，把学生网络的RPN的预测框和教师网络的高质量边框 $B_p$ 放在一起，得到 $P^* = RPN_{student}(I_t) + B_p$
  - $P^*$ 之后被送到学生网络的 ROI 模块来产生最终预测的分类分数和回归边框 $C_{student} ， B_{student}$
  
- Phase 2 :
  - 教师网络使用与学生网络相同的拼接的 $P^*$ ，通过教师网络的 ROI 模块，同样产生最终的 $C_p^*, B_p^*$ 
  
  - 由于教师网络最终的输出和学生网络最终的输出都来自于同样的 $P^*$ ，因此可以相互组成匹配对。

  - 如果直接使用教师网络产生的伪标签，要么会获得都是简单的标签，要么会产生错误标签。因此本文没有直接用教师网络的伪标签来训练学生网络，而是给定同样一组 $P^*$ ，让学生网络的预测尽可能接近教师模型的预测，即使某个预测框本身就是错的，但是训练的目标是使两个模型的预测分布接近，并没把教师模型的伪标签当成硬标签直接让学生模型学习。
  
  - 然而，让学生模型不断学习教师模型的预测分布，也会一定程度上存在错误累积，因此本文对KL散度乘了一个权重 $\alpha$ ：
    $$
    L_{cons} = \alpha KL(C_{student}, C_p^*)
    $$
    对于 $P^*$ 中的第 $i$ 个 region proposal， $\alpha_i = max(C_p^*)$ ，表示第 $i$ 个 region proposal 的KL散度的权重系数是该 region proposal 的预测概率最大值。 

  最终的 loss 由白天图像（经过数据增强）和无标签夜晚图像的两部分损失组成：
  $$
  L_{total} = L_{sup} + \lambda L_{unsup}
  $$
  其中，
  $$
  L_{sup} = L_{rpn}(B_s, I_s) + L_{roi} (B_s, C_s, I_s)
  $$
  是学生模型在有标签的白天图像的two-stage损失
  $$
  L_{unsup} = L_{rpn}^{obj}(C_p^*; I_t) + L_{cons}(C_p^*; I_t)
  $$
  教师模型在有标签的白天图像只计算RPN的损失 $L_{rpn}^{obj}(C_p^*; I_t)$ ，$L_{cons}(C_p^*; I_t)$ 是KL散度蒸馏损失。
  
  实验中，$\lambda = 0.3$
  
  在预训练之后，两阶段训练还需要训练额外的 50K iters。学习率保持 0.04 不变，bs=3白天+3夜晚=6
  
  
  ## 2.2 Student-Scaling
  
  根据调查，对夜晚目标检测影响最大的是小目标。这是由于小目标的特征在夜晚黑暗场景中更容易受到模糊和噪声的影响而造成信息损失。为了克服这个问题：
  
  - 在学生模型的预训练阶段，作者使用了尺度缩放的数据增强方法：通过一种策略来不断增加输入图像的尺度，直至原始输入尺度。即，刚开始训练的图像较小，能够产生更多的小目标，使得学生模型在预训练阶段能够更多关注小目标。具体的，原始输入尺度为resize最短边至640，在预训练阶段的第百分之 $ (0.57, 0.64,0.71, 0.78, 0.85, 0.92) $ 个 iters，使用的缩放尺度为 $(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)$ 。对 BDD100K数据集和SHIFT数据集，分别预训练 50k 和 20k 个iters。
  - 尺度缩放的数据增强能够使后续的教师模型对小目标的预测更准确。
  - 为了避免小尺度训练时，有些目标过小从而变成噪声标签，作者设置了一个最小尺度的阈值来移除这些物体。
  - 预训练时从较小的输入尺度逐渐增长至目标分辨率，在之后的教师-学生模型训练过程中，怎么避免遗忘之前小尺度学习的知识？作者使用了一个高斯函数来采样尺度缩放因子，未详细介绍，但应该是：以较大的概率采样目标输入尺度，以较小的概率采样小尺度。
  - 预训练阶段不使用 NightAug
  
## 2.3 NightAug

NightAug时作者提出的一个针对夜晚的数据增强pipeline，无需其他领域自适应方法的训练过程。NightAug由一些数据增强方法组成，目的是抑制白天的特征。

作者认为夜晚图像的特点：

- darker
- 低对比度
- 由于数字相机导致的更高的 signal-to-night ratio (SNR)，比如亮度和彩噪
- 可能存在灯光炫光
- 由于夜晚成像不清，相机拍摄的图像可能失焦

NightAug包括：

- 随机亮度
- 随机对比度
- 随机gamma
- 随机高斯噪声
- 随机高斯模糊
- 随机插入炫光

此外，为了进一步增加图像的多样性，在每种数据增强的过程中，都会随机的选择并忽略图像中的一部分区域，这种方法能够保证图像的各部分不是均匀的亮度。

NightAug的算法流程如下图所示：

![image-20230418221204438](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418221204438.png)

伪代码如下：

![image-20230418221925938](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418221925938.png)

数据增强的效果如下图所示：

![image-20230418221950960](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418221950960.png)

# 3 实验

## 3.1 消融实验

![image-20230418224126484](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418224126484.png)

![image-20230418224334874](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418224334874.png)

上图可以看出，如果加上领域分类器，效果会显著下降（可能是作者做了数据增强，领域分类器无法正常学习？）。

## 3.2 结果对比

![image-20230418224153225](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418224153225.png)

![image-20230418224231758](imgs/28-Two-Phase%20Consistency%20Training%20for%20Day-to-NightUnsupervised%20Domain%20Adaptive%20Object%20Detection/image-20230418224231758.png)
