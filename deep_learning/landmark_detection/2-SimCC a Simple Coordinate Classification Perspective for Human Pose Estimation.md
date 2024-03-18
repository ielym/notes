# 1 介绍

基于 Heatmap 的方法面临量化误差问题，这是由于需要把连续的坐标映射到离散的下采样层导致的，进而产生几种的缺点：

- 为了减轻量化误差的影响，一些方法通常使用额外的上采样层，如反卷积来增加特征图的分辨率。但是产生了额外的计算开销。
- 或者使用额外的后处理来改进预测值，如 DARK
- 在分辨率较低的特征图上，效果远不足以满足要求

因此，本文提出了一种简单但是有效的坐标分类pipeline，称为SimCC。该方法把关键点估计看作一个对水平坐标和水平坐标的分类方法：

- 首先使用CNN or Transformer 等backbone来提取关键点特征。
- SimCC对水平坐标和垂直坐标进行独立的分类，来获得最终的预测。
- 为了减少量化误差，SimCC均匀的把特征图的每个像素点划分成若干个更小的bins，从而实现亚像素级别的定位精度。
- heatmap-based方法通常需要引入多个反卷积层，而SimCC只需要两个轻量级的头，每个头都只有一个线性层。

与基于heatmap的2D方法比较，SimCC的有点包括：

- 通过把每个像素均匀的划分成若干个bins的方法，能够减少量化误差
- SimCC省去了上采样层，并且不包含计算量较大的后处理
- SimCC即使在低输入尺寸的情况下，依然具有较强的性能

# 2 SimCC

SimCC的核心思想是把关键点估计任务替换为两个分类任务，一个对水平方向，一个对垂直方向。并且通过把每个像素划分成多个bins来减少量化误差。

![image-20230421214844936](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421214844936.png)

Backbone : 对于尺寸为 $H \times W \times 3$ 的输入图像，SimCC使用CNN或Transformer作为backboen来提取n个关键点的特征。

## 2.1 Head

水平和垂直的两个分类器在backbone之后，用于坐标分类。对于CNN-based方法，本文把n个关键点的特征图 $(n, H', W')$ 直接 flatten 成 $(n, H' \times W')$  ，并用于分类。相较于使用多个反卷积的heatmap-based的方法，该方法更轻量。

## 2.2 坐标分类

对于每个关键点，水平和垂直分类器的输出维度分别为 $N_x$ 和 $N_y$ 。即，水平方向和垂直方向的分类器各预测 $N_x$ 个类别和 $N_y$ 个类别。即，表示把特征图分成沿着水平和垂直方向分别划分成 $N_x$ 个bins和 $N_y$ 个bins。

其中， $N_x = k \cdot W$ ， $N_y = k \cdot H$ 。最终，关键点的gt在水平和垂直方向上分别被指定到某个bins中：$c_x \in [1, N_x]$ ，$c_y \in [1, N_y]$ 。模型预测的第 $i$ 个关键点的bins索引为 $o_x^i, o_y^i$ 。最终，使用KL散度作为损失函数。

其中，$k \ge 1$ ，用于产生大于等于原始输入尺寸的bins，从而减少量化误差，获得亚像素级别的定位精度。 

## 2.3 Label Smoothing

标签平滑在图像分类任务中被广泛使用，来增强模型性能。因此，本文也采用标签平滑，称为  equal label smoothing。

然而，一般的Label Smoothing对负样本位置使用同样的smooth label，缺乏关键点空间位置比较接近的位置的相关性：

- 输出类别（bins的索引）距离gt越近越好
- 使用拉普拉斯或高斯标签平滑

## 2.4 与2D Heatmap方法对比

### 2.4.1 Quantization error

由于在高分辨率的特征图上计算Heatmap的计算开销较大，因此 2D heatmap-based 的方法通常使用的是下采样 $\lambda \times $ 的特征图，会产量量化误差。而 SimCC 把每个像素划分成 $k \ge 1$ 的bins，能够有效减少量化误差，并获得亚像素级别的定位精度。

## 2.4.2 Refinement Post-Processing

Heatmap-based方法为了减少量化误差，通常需要依赖于繁重的额外后处理。如果没有这些后处理，heatmap-based方法的性能会明显下降：

![image-20230421230010574](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421230010574.png)

可以看出，输入图像分辨率越低，在没有后处理是的性能下降越大。

### 2.4.3 Low/high Resolution Robustness

![image-20230421230154666](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421230154666.png)

如上图所示，在任意输入图像分辨率下，SimCC都稳定的优于heamap-based方法，特别是在低分辨率的情况下，增益更明显。

# 3 实验

## 3.1 Splitting factor k

k 表示把每个像素分成多少个bins，越大的k就能够获得越小的量化误差，然而，模型的训练也会变得更困难。本文测试了 $k \in \{1, 2, 3, 4\}$ 。结果如下图所示：

![image-20230421230623104](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421230623104.png)

## 3.2 上采样模块

![image-20230421230736015](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421230736015.png)

如上图所示，在没有反卷积上采样层的情况下，SimCC的性能几乎不变，节省了很多计算开销。

## 3.3 Label Smoothing

![image-20230421230823685](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421230823685.png)

如上图所示，equal表示普通的标签平滑（负样本的各个位置都相等），以及高斯，拉普拉斯和不使用标签平滑的方法。可以看出使用标签平滑能够有效提升AP。

## 3.4 密集场景

同样比Heatmap-based方法好：

![image-20230421231007422](imgs/2-SimCC%20a%20Simple%20Coordinate%20Classification%20Perspective%20for%20Human%20Pose%20Estimation/image-20230421231007422.png)