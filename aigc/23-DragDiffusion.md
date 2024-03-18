`DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing`



# Introduction

DragGAN的缺点：

- 基于GAN的生成能力上限有限

因此本文使用扩散模型实现Drag的功能：

- 整体流程与DragGAN相同，包括两个连续的过程：运动监督和点跟踪
- DragGAN需要进行多次迭代，并生成多次图像来进行点集的运动。在Diffusion中每生成一张都需要t-steps，如果再生成多张图像，耗时和计算量是不可接受的。有趣的是，作者发现，不必多次生成图像，只需要在diffusion隐编码空间，通过一定步数的监督运动过程就可以进行精确的空间编辑（**一定步数指的是在原图加噪的隐编码上进行的，而不是去噪过程**）。
- Diffusion由于不断进行去噪，随机性较大，作者发现在dragging的过程中，物体的外观很容易发生变化，导致和原始图像不匹配（如猫头变成狗头，图像整体风格变化）。为了解决这个问题，在编辑图像之前，都需要首先针对该图像过拟合一个LoRA，来保证重建过程的稳定性。

# Method Overvies

![image-20230804110800303](imgs/23-DragDiffusion/image-20230804110800303.png)

整体流程分成三步：

- 使用图像训练一个LoRA，用于更好的固定住物体和风格。
- 使用DDIM的逆向过程来获取Diffusion的初始隐编码。并在隐编码上重复进行运动监督和点跟踪来优化初始隐编码。
- 把编辑好的初始隐编码送到DDIM中，产生最终的结果。

## Motion Supervision and Point Tracking

## Motion Supervision

定义：

- 输入图像 $z_0$ ，对其进行DDIM逆向加噪，得到第 $t$ 个时间步的隐编码 $z_t$ 。对 $z_t$ 进行多次运动达到目标隐编码 $\bar{z_t}$
- 共有 $n$ 个待运动的 handle points，其中第 $i$ 个点 $h_i$ 在第 $k$ 次运动时的位置为 $\{h_i^k = (x_i^k, y_i^k)\}$ ，$i \in \{1, ..., n\}$ ，$k$ 根据实际停止条件确定，不固定。
- $h_i^k$ 周围 $r_1$ 范围内的点集定义为 $\Omega (h_i^k, r_1) = \{ (x, y) : |x-x_i^k| \le r_1,  |y-y_i^k| \le r_1 \}$ ，是一个变成为 $2 r_1 +1$ ，以 $\{h_i^k = (x_i^k, y_i^k)\}$ 为中心的方形区域。
- 待运动的目标点集为 $\{g_i = (\bar{x_i}, \bar{y_i})\}$ ，$i \in \{1, ..., n\}$

- 对于隐编码 $z_t$ ，输入到Unet，在Unet的倒数第二层的特征定义为 $F(z_t)$ ，使用 $F(z_t)$ 来进行运动监督。
- 特征 $F(z_t)$ 上位置 $h_i^k$ 处的特征为 $F_{h_i^k} (z_t)$ 
- handle point每次运动的归一化距离为 $d_i = (g_i - h_i^k) / ||g_i - h_i^k||_2$ 

运动监督损失为：
$$
L(\bar{z}_t^k) = \sum_{i=1}^n \sum_{q \in \Omega(h_i^k, r_1)}
|| F_{q+d_i} (\bar{z}_t^k) -  sg(F_{q} (\bar{z}_t^k))||_1

+ \\

\lambda ||  (\bar{z}_{t-1}^k - sg(\bar{z}_{t-1}^0) \odot (\mathbb{1} - M) ||_1
$$
 其中：

- $\bar{z}_t^k$ 是原图加噪的 $z_t$ 经过 $k$ 次运动后的隐编码。
- $sg(\cdot)$ 表示  stop gradient
- 由于 $F_{q+d_i}$ 不是整型的位置，因此使用双线性插值。
- $M$ 是用户定义的 binary mask，$M$ 之外的地方不变。

需要注意，对于运动监督，是在Unet的倒数第二层输出特征上进行的。而Mask控制不变是在隐编码上进行的：

- 对于 $\bar{z}_t^k$ ，首先使用 DDIM 进行一次去噪过程，获得 $\bar{z}_{t-1}^k$ 。让其与原始图像的隐编码经过一次DDIM去噪过程的 $\bar{z}_{t-1}^0$ 保持一致。

梯度更新：

- 经过上述Loss，更新 $\bar{z}_{t}^k$ ，得到 $\bar{z}_t^{k+1}$ ：
  $$
  \bar{z}_t^{k+1} = \bar{z}_t^{k} - \eta \frac{\partial L(\bar{z}_t^k)}{\partial \bar{z}_t^k}
  $$
  $\eta$ 是学习率

## Point Tracking

由于运动监督更新了 $\bar{z}_t^k$ ，handle points 的位置可能发生了变化。因此，我们需要通过点跟踪来更新 handle point。

受 `Emergent correspondencefrom image diffusion` 的启发，作证证明在Unet的特征上进行点跟踪，也能够对应上在原图上的位置。因此，本文使用 $F(\bar{z}_t^{k+1})$ 和 $F(\bar{z}_t)$ 进行点跟踪：

- 对于 handle point $h_i^k$ ，其最近邻的点集为 $\Omega(h_i^k, r_2) = \{ (x, y) : |x - x_i^k| \le r_2, |y - y_i^k| \le r_ \}$ 

- $h_i^k$ 在经过运动监督之后，新的位置 $h_i^{k+1}$ 为：
  $$
  h_i^{k+1} = argmin_{q \in \Omega(h_i^k, r_2)} || F_q(\bar{z}_t^{k+1} - F_{h_i^k} (z_t)) ||_1
  $$

# 实验

- 底模：SD1.5
- 代码 ： diffusers
- 使用LoRA微调 q, k, v， rank = 16 ，AdamW 优化器，学习率 2e-4，训练200个steps实现计算时间和质量的平衡
- 编辑阶段，DDIM的采样步数设置为 50 步。
- 优化 t=40 步的diffusion的隐编码。学习率为0.01。
- 在 DDIM 的 inversion 和 denoising 阶段，都没有使用 CFG，因为CFG会扩大数值误差。
- $r_1 = 1, r_2 = 3, \lambda=0.1$ 。
- text prompt 需要保证和微调 LoRA 时的相同。