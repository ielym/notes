`SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds`



# 1 Motivation

- SD模型非常大，有复杂的网络结构并需要数十次迭代去噪。在云端跑 SD 的推理涉及到隐私问题，特别是用户数据发送给第三方时。
- 目前加快推理速度的方法都是基于量化，针对GPU优化等。但这些方法的速度仍然不能满足用户的无缝体验（三星S23 Ultra上 11.5s）。还有一些post-training的优化，如pruning, NAS，减少参数量等，然而这些方法如果不经过大量的微调，很难达到原始效果。此外，直接减少采样步长的方法通常影响通用的性能，而渐进式蒸馏的方式可以缓解这个问题。
- 本文提出了一个通用的方法，通过提出一个高效的UNet，以及step蒸馏，实现在移动设备上低于2s的推理。

# 2 SD模型分析

## 2.1 前置知识

扩散模型的优化目标为：
$$
min \mathbb{E}_{t \sim U[0, 1], x\sim p_{data}(x), \epsilon \sim N(0, I)} ||\hat{\epsilon}_\theta(t, z_t) - \epsilon||_2^2
$$
其中，$t$ 是时间不，$\epsilon$ 是gt的噪声，$z_t = \alpha_t x + \sigma_t \epsilon$ 是噪声数据输入，$\alpha_t, \sigma_t$ 是噪声信号的强度（从$x_0$ 加噪成step t 的噪声），$\epsilon_\theta(\cdot)$ 是扩散模型，参数是 $\theta$ 

在 DDIM 中，定义 $t' \lt t$ ，有：
$$
z_{t'} = \alpha_{t'} \frac{z_t - \sigma_t \hat{\epsilon}_\theta(t, z_t)}{\alpha_t} + \sigma_{t'} \hat{\epsilon}_\theta(t, z_t)
$$
 之后，$z_{t'}$ 再喂给 $\hat{\epsilon}_\theta(\cdot)$ ，直到 $t'$ 变成0。

---

SD模型为了降低计算量，在隐空间进行扩散过程，推理时使用VAE的解码器解码。LDM (latent diffusion model) 也支持文本编码输入 $c$ ，当合成图像时，CFG通常用来改善生成质量：
$$
\tilde{\epsilon}_\theta(t, z_t, c) = w\hat{\epsilon}_\theta(t, z_t, c) - (w - 1) \hat{\epsilon}_\theta(t, z_t, \empty)
$$
其中，$\hat{\epsilon}_\theta(t, z_t, \empty)$ 表示无条件输出。guidance scale w可以用于调整条件信息的强度，来实现质量和多样性的平衡。

## 2.2 SD1.5基线模型分析

![image-20230725145157718](imgs/19-SnapFusion/image-20230725145157718.png)

![image-20230725145203822](imgs/19-SnapFusion/image-20230725145203822.png)

如上所示，SD通常包含三个部分：

- Text Encoder ：生成一张图像，需要运行两次（一次for CFG），时延只有 8ms，非常轻量。
- VAE解码器输入隐特征，输出图像，通常需要369ms。比较大，但不是最主要的问题。
- UNet运行一次的时延时就有 1.7s，而且通常需要运行数十次，是总时延的大头。

---

SD中的UNet主要包含 Cross-Attention 和 ResNet blocks，两部分的参数两和计算量如下图所示：

![image-20230725145805024](imgs/19-SnapFusion/image-20230725145805024.png)

上图说明：

- 参数量最大：发生在Middle Stage的 ResNet中，这是由于通道维度逐渐增大导致的。
- 计算量最慢：发生在下次啊杨的输入输出阶段，这是由于图像的空间分辨率最大，进行空间维度的Cross/Self注意力机制时，计算复杂度和空间分辨率呈平方的增长。

# 3 结构优化

## 3.1 高效UNet

通过实验观察，在使用剪枝或搜索导致算子改变时，生成图像的性能有明显退化，需要显著的训练消耗来恢复性能。因此我们提出了一种鲁棒性的训练方式。

### 3.1.1 Robust Training

我们使用随机前向传播来执行每个 cross-attention 和 ResNet block，概率为 $p(\cdot, I)$ 。其中，$I$ 表示 identity mapping（跳过对应的 $\cdot$  操作），$\cdot$ 表示 cross-attention 或 ResNet block。$p(\cdot, I)$ 表示以一定的概率，或者执行 $\cdot$ ，或者执行 $I$ 。

通过这种训练增强，网络对不同的网络组合会变得鲁棒。使得我们能够准确的评估不同block的贡献。

### 3.1.2 Evaluation and Architecture Evolving 

使用 Robust Training 训练好之后的网络，可以保证在移除任一block之后，网络整体的性能退化不会太大。因此，为了分析不同block的贡献程度，作者采样随机移除/添加block的方式进行对比：

- $A_{Cross-Attention[i,j]}^{+,-}$ 和 $A_{ResNet[i,j]}^{+,-}$ 表示移除 ($-$) 一个block，或增加 ($+$) 一个block，$i,j$ 表示第 $i$ 个 stage 的 第 $j$ 个block
- 对于每个block的分析，都要同时考虑速度和效果。因此，每移除一个block，都使用 $\Delta CLIP / \Delta Latency$ 来作为评价指标，测试整个模型的得分。得分越高，表示该block的贡献越低。
- 测试数据是应用 MS-COOC的一个小的子集（约 2K 图像），固定步长为50，CFG为7.5。评测一次需要花费 2.5 A100 GPU 小时。

具体评测算法如下：

![image-20230725154529237](imgs/19-SnapFusion/image-20230725154529237.png)

最终的网络结构如下：

![image-20230725154654038](imgs/19-SnapFusion/image-20230725154654038.png)

# 4 实验

![image-20230725154752659](imgs/19-SnapFusion/image-20230725154752659.png)

为了验证鲁棒训练的有效性，作者做了一组对比实验，如上图所示：

- 图 b 表示原始SD1.5移除了 CA 之后，直接就崩了。
- 而经过鲁棒训练之后，无论是移除 CA 还是移除 RB，效果都不至于崩。