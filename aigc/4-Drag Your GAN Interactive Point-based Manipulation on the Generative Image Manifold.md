Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold



# 2 背景

## 2.1 StyleGAN2

- StyleGAN2 有一个 Mapping Network，用于把 $512-d$ 的隐编码 $z \sim N(0, I)$ 映射成中间层级的隐编码 $w \in \mathbb{R}^{512}$ 。$w$ 的隐编码空间称为 $W$ 。

- 之后，$w$ 送到生成器 $G$ 来产生输出图像 $I = G(w)$ 。在这个过程中，$w$ 会输入给 $G$ 的多个不同的网络层，来更好的控制 $G$ 中不同网络层的属性。如下图所示：

  ![image-20230530163528160](imgs/4-Drag%20Your%20GAN%20Interactive%20Point-based%20Manipulation%20on%20the%20Generative%20Image%20Manifold/image-20230530163528160.png)

- 此外，也可以给不同的网络层输入不同的 $w$ ，假设一共给 $l$ 个网络层输入 $w$ ，则总的 $w$ 的维度就是 $w \in \mathbb{R}^{l \times 512} = W^{+}$ 。显然 $W^+$ 比 $W$  获取的成本更高。  

## 2.2 Point-based Manipulation

![image-20230530163848103](imgs/4-Drag%20Your%20GAN%20Interactive%20Point-based%20Manipulation%20on%20the%20Generative%20Image%20Manifold/image-20230530163848103.png)

图像manipulation的基本方法如上图所示：

- 通过隐编码 $w$  生成的图像 $I \in \mathbb{R}^{3 \times H \times W}$ ，用户可以输入一些原始的点 $\{p_i = (x_{p,i}, y_{p,i}) | i = 1,2,3,4,...,n\}$ 以及这些点对应的目标位置的点  $\{t_i = (x_{t,i}, y_{t,i}) | i = 1,2,3,4,...,n\}$ 。
- Point-based Manipulation 的目标是移动图像中的物体，让其从原始位置移动到目标位置，并保持语义。

# 3 方法

如上图所示，本文的方法包括两个子阶段：

- Motion Supervision ：通过计算loss，强制让原始点运动到目标点的位置。该loss用于优化隐编码 $w$ 。在一次优化完成之后，我们获得了一个新的隐编码 $w'$ 和一张新的图像 $I'$ 。每次更新只对图像做轻微的改变。
- Point Tracking ：一次移动完成之后，目标点的位置不变，但此时原始点 $\{p_i\}$ 的位置已经发生了移动，因此需要更新 $\{p_i\}$ 的位置。Tracking 是非常重要的，因为如果原始点在移动之后不进行跟踪，下一次移动的可能就是错误的点了（如第一次是鼻子，但移动之后的 $p_i$ 不准确的话，下一次可能移动的就是脸部了）。
- 跟踪之后，在新的 $\{p_i\}$ 和新的隐变量 $w'$ 上重复上述的过程。知道 $\{p_i\}$ 到达 $\{t_i\}$ 位置。该过程通常需要 30-200次迭代。

## 3.1 Motion Supervision

整个移动的过程是在特征图上进行的。作者认为：

- 生成器中间隐层的特征已经足够具有判别性，因此使用简单的loss就足以监督运动的过程
- StyleGAN2的第 6 个隐层的特征在分辨率和判别性上由很好的trade-off，因此选用该层的特征 $F$ 来进行移动。

具体流程如下：

- 首先，把 $F$ 通过双线性插值 resize 到最终输出图像的尺寸

- 为了把点 $p_i$ 移动到点 $t_i$ ，本文的思想是不仅监督 $p_i$ 移动到 $t_i$ ，还监督 $p_i$ 周围 $r_1$ 半径内的所有点都按照 $p_i \to t_i$ 的向量进行移动。如下图红色圆圈 和 蓝色圆圈所示。其中 $p_i$ 周围半径 $r_1$ 范围内所有点的集合记作 $\Omega_1(p_i, r_1)$

  ![image-20230530165910729](imgs/4-Drag%20Your%20GAN%20Interactive%20Point-based%20Manipulation%20on%20the%20Generative%20Image%20Manifold/image-20230530165910729.png)

- 移动前后，需要对齐红色圆圈和蓝色圆圈内所有点的像素值:
  $$
  \sum_{q_i \in \Omega_1(p_i, r_1)}
  ||
  F(q_i) - F(q_i + d_i)
  ||_1
  $$
  其中，$d_i$ 为移动的归一化的向量：
  $$
  d_i = \frac{t_i - q_i}{||t_i - q_i||_2}
  $$
  然而，$q_i$ 在移动 $d_i$ 距离之后，可能不是整数坐标，文中采用的是双线性插值的方法。该项loss的伪代码为：
  $$
  l1 (blue, red.detach())
  $$
  也就是论文中的

  ![image-20230530173522194](imgs/4-Drag%20Your%20GAN%20Interactive%20Point-based%20Manipulation%20on%20the%20Generative%20Image%20Manifold/image-20230530173522194.png)

- 此外，如果给定 mask $M$ ，则需要保证 $M$ 之外的图像保持不变，即：
  $$
  \lambda 
  ||
  (F - F_0) \cdot (1 - M)
  ||_1
  $$
  其中，$F_0$ 是最最最早的原始图像，在迭代过程中， $F_0$ 始终不变。

最终的 Motion Supervision 的 Loss 为：
$$
L = \sum_{i=0}^{n}

\{
\sum_{q_i \in \Omega_1(p_i, r_1)}
||
F(q_i) - F(q_i + d_i)
||_1
+

\lambda 
||
(F - F_0) \cdot (1 - M)
||_1
\}
$$
需要注意，虽然计算了 Loss，但是该 Loss 不更新网络的权重，只更新 $w$ ，类似于对抗攻击中的修改输入图像，给图像加高频噪声的方式。

为了获得更好的可编辑性，本文使用 $W^+$ 空间，即，每层的 $w$ 单独更新。实验发现，对于空间属性，前6层受 $w$ 的影响较大，而其他层只改变外观属性。因此，受 style-mixing 的启发，实验中只改变前6层的 $w$ ，其他层的 $w$ 保持不变。

 

## 3.2 Point Tracking

![image-20230530185921408](imgs/4-Drag%20Your%20GAN%20Interactive%20Point-based%20Manipulation%20on%20the%20Generative%20Image%20Manifold/image-20230530185921408.png)

在 motion supervision 中，通过更新 $w$ 的梯度，得到了新的隐编码 $w'$ ，新的特征图 $F'$ 和新的图像 $I'$ 。

由于 $q_i + d_i$ 是浮点数，在 motion supervision 中无法准确定位源坐标，因此在 Point Tracking 中需要跟踪对应的点。

Point Tracking 通常使用光流估计模型 或 particle Video approaches [Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories] 方法。然而这些方法存在效率较低和累积误差的问题。因此，本文提出了一中新的 Point Tracking 方法：

- 定义初始图像上 $p_i$ 位置的特征为 $f_i = F_0(p_i)$ 

- 定义$p_i$ 周围的点集为 $\Omega_2(p_i, r_2) = \{ (x, y) | |x - x_{p,i}| < r_2, |y - y_{p,i}| < r_2 \}$ ，如下图所示：

  ![image-20230530184852000](imgs/4-Drag%20Your%20GAN%20Interactive%20Point-based%20Manipulation%20on%20the%20Generative%20Image%20Manifold/image-20230530184852000.png)

- 对于每次迭代的特征图 $F'$ 上点集中的点 $q_i$ ，计算其与 $f_i$ 的L1距离，取距离最小的作为匹配到的点，并更新 $p_i$ ：
  $$
  p_i := argmin_{q_i \in \Omega_2(p_i, r_2)} ||F'(q_i) - f_i||_1
  $$

- 如果存在多个handle point起始点，对每个点分别使用上述方法计算跟踪点。

- Point Tracking 使用的$F'$ 同样来自于 StyleGAN2 的第 6 个 block的特征，分辨率为 $256 \times 256$ 。可以也可以不resize到输入图像分辨率。

# 4 实现细节

- 使用 Adam 优化器来优化 $w$ ，$lr=2e-3$ for FFHQ数据集，AFH!QCat数据集和LSUN Car数据集。其他数据集的学习率为 $1e-3$ 。
- $\lambda = 20, r_1 = 3, r_2 = 12$ 
- 当 handle points 的数量小于等于5时：当所有的 handle points 都和其对应的 target points 的距离小于1时，停止迭代。
- 当 handle points 的数量大于5时：当所有的 handle points 都和其对应的 target points 的距离小于2时，停止迭代。