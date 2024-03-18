# Adversarial Diffusion Distillation

![image-20231205222925196](imgs/37-SDXL_Turbo/image-20231205222925196.png)

# 1 引用

- 论文：https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf

# 2 介绍

提出了对抗扩散蒸馏 （Adversarial Diffusion Distillation, ADD），一种新的训练方法，可以在1-4step中采样出高质量的图像。实验证明在单步的情况下优于GANs和LCM模型，只需要4步能够超过SDXL。

- 扩散模型能够处理如图像生成等复杂任务，但是，迭代推理过程需要多个采样步，无法实时。
- GANs只需要单步，但是尽管扩展到大型数据集，采样质量还是不如扩散模型

- 因此本文的工作是结合二者的优点。

结合了两个训练目标：

- **adversarial loss 对抗损失** ：在每个生成时间步中强迫模型直接生成真实图像manifold的样本，避免出现其他蒸馏方法存在的模糊和伪影问题。
- **score distillation sampling （SDS）分数蒸馏采样：**使用另外一个预训练并且固定的扩散模型作为教师，有效的利用预训练模型的大量知识，并保留大型扩散模型中的强组合性。

- 推理时，该方法不使用CFG，进一步降低内存消耗。

# 3 方法

![image-20231205230848391](imgs/37-SDXL_Turbo/image-20231205230848391.png)

## 3.1 训练过程

训练过程如上图所示，包含三个网络：

- 一个ADD-Student网络，从预训练的UNet-DM中初始化，参数为 $\theta$ 
- 一个可训练的判别器，参数为 $\phi$
- 一个DM的教师模型 DM-Teacher，冻结参数，参数为 $\psi$

训练流程：

-  真实输入图像 $x_0$ ，对学生模型用均匀分布从 $T_{student} = \{ \tau_1, ...,\tau_n \}$ 中采样出一个时间步 $s$ 。其中$N=4$ ，表示蒸馏4步推理的学生模型。并且 $\tau_n = 1000$ ，因为必须要保证模型从纯高斯噪声（$SNR=0$） 开始去噪。用学生模型ADD-Student的参数加噪得到 $x_s = \alpha_s x_0 + \sigma_s \epsilon$ 。之后把噪声图像 $x_s$ 送给 ADD-Student模型，得到预测图像 $\hat{x}_\theta(x_s, s)$ 。

- 计算对抗损失：

  - 学生模型生成的图像 $\hat{x}_\theta$ 和真实图像 $x_0$ 送到判别器中计算对抗损失，具体细节下节介绍。
  - 计算对抗损失 $L_{adv}^G (\hat{x}_\theta(x_s, s), \phi)$ ，$\phi$ 应该表示空集，因为判别损失没有 gt 。

- 计算蒸馏损失：

  - 对于学生模型生成的图像 $\hat{x}_\theta$ ，对教师模型采样出一个时间步 $t \in T_{teacher} = \{ 1, 2, ..., 1000\}$ （暂时没看到 $t$ 和 $s$ 的对应关系计算，也没有提到采样 $t$ 的分布）。用教师模型 DM-Teacher的参数加噪得到噪声图像 $\hat{x}_{\theta, t}$ ，并用教师模型去噪得到预测图像 $\hat{x}_{\psi}(\hat{x}_{\theta, t}, t)$ 。
  - 计算蒸馏损失 $\lambda L_{distill} (\hat{x}_\theta(x_s, s), \psi)$

- 计算整体损失：
  $$
  L = L_{adv}^G (\hat{x}_\theta(x_s, s), \phi) + \lambda L_{distill} (\hat{x}_\theta(x_s, s), \psi)
  $$

上述方法是在像素空间进行的。如果教师模型和学生模型的latent space是共享的（VAE相同），上述过程也可以在 latent space进行。作者认为在像素空间计算损失时，蒸馏损失的梯度会更稳定。

## 3.2 Adversarial Loss

![image-20231211111957740](./imgs/37-SDXL_Turbo/image-20231211111957740.png)

判别器按照 StyleGAN-T的设计和训练方法，如上图最右侧所示（图来自于StyleGAN-T论文）：

- 有一个冻结的预训练的特征提取网络 $F$ ，如上图的 DINO Encoder：
  - StyleGAN-T论文中，DINO Encoder使用的是 ViT-S，使用自监督的DINO目标训练。该网络是轻量快速的，能够在高分辨率中编码语义信息。
- 有一系列的 Discriminator head $D_{\phi, k}$ ，每个判别器的输入特征是 $F_k$ 。判别头是轻量的，具有可训练参数。结构如上图所示，其中 Conv1 是 1D卷积：
  - 由于ViT架构中，不同深度的网络层的输出维度 $tokens \times channels$ 都是相同的，因此可以在网络不同深度的地方使用相同架构的 Discriminator head 
  - 多个 Head 的效果更好，StyleGAN-T在网络中均匀的使用了 5 个 heads 
  - 如 StyleGAN-T上图的右下角所示，残差卷积核（1D）的kernel width控制了 Discriminator head在 token sequence 上的感受野大小。
  - StyleGAN-T发现，在 token 上直接使用 1D卷积的效果，和在 reshaped tokens 上使用 2D卷积的效果相比，使用2D卷积并不会有任何好处。表明判别器并不会从tokens中保留任何2D结构。
- 为了改善性能，判别器可以通过projection加入额外的condition信息，如上图右下角的 $Affine (c_{text})$  ：
  - 但是在 Turbo 中，与StyleGAN-T不同的是，Turbo可以使用图像作为条件。
  - 对于学生模型的 $\tau < 1000$ ，ADD-student能够从 $x_0$ 中获取部分信号。因此，判别器也可以利用 $x_0$ 的信息，对生成图像 $\hat{x}_\theta(x_s, s)$ 进行更有效的判别 （生成图像利用了 $x_0$ 的信息 ，判别器也利用 $x_0$ 的信息）。实现上，Turbo 使用了一个额外的特在提取网络，对原始图像 $x_0$ 提取特征编码 $c_{img}$ 。
  - 判别器同时用了 $c_{img}$ 和 $c_{text}$ 特征

判别器的损失使用 Hinge Loss ：
$$
L = max(0, 1 - y \cdot \hat{y})
$$
当预测标签与真实标签同号时，loss是0。

 Turbo 中的 $L_{adv}(\hat{x}_\theta(x_s, s), \phi)$ ：
$$
L_{adv}^G (\hat{x}_\theta(x_s, s), \phi) = - E_{s,\epsilon, x_0} [\sum_k  D_{\phi,k} (F_k(\hat{x}_\theta(x_s, s)))]
$$
即，$k$ 个 DINO 的特征判别器的输出越大越好。判别器训练时需要最小化：
$$
L_{adv}^D (\hat{x}_\theta(x_s, s), \phi) = E_{x_0} [\sum_k  max(0, 1 - D_{\phi,k} (F_k(x_0) + \gamma R1(\phi)]

\\+

E_{\hat{x}_0} [\sum_k  max(0, 1 + D_{\phi,k} (F_k(\hat{x}_\theta)]
$$

- 真实图像 $x_0$ 需要判别器的预测值越大越好
- 生成图像 $\hat{x}_\theta$ 需要判别器的预测值越小越好
- $R1$ 表示 R1 gradient penalty 。 `Rather than computing the gradient penalty with respect to the pixel values, we compute it on the input of each discriminator head Dϕ,k. We find that the R1 penalty is particularly beneficial when training at output resolutions larger than 1282 px.`

---

对于判别损失的理解，先从GAN的角度理解比较简单，即只使用单层特征 $F$ ，且原始GAN的loss和上式有一定差别：
$$
L = E_{x \in p(x)} [-log D(x)] + E_{z \in p(z)} [-log (1 - D(g(z)))]

\\=

\int_x p_{data}(x) log(D(x)) dx + \int_z p(z) log(1 - D(g(z))) dz
$$
其中，$x$ 是真实样本，$z$ 是隐编码，$g(z)$ 是生成器生成图像，$p_{data}(x)$ 是真实图像的数据分布，$p(z) = p_g$ 是生成器生成图像的数据分布。

根据 https://zhuanlan.zhihu.com/p/53141065 介绍，根据 Radon-Nikodym定理，可以对上式进行换元：
$$
\int_z p(z) log(1 - D(g(z))) dz = \int_{p_g(x)} p_g(x) log(1 - D(x)) dx
$$
因此：
$$
L = \int_x p_{data}(x) log(D(x)) dx + \int_{p_g(x)} p_g(x) log(1 - D(x)) dx

\\=

\int_x [p_{data}(x) log(D(x)) dx + p_g(x) log(1 - D(x))] dx
$$
求 $L$ 的极值：
$$
p_{data} \frac{1}{D(x)} - p_g \frac{1}{1 - D(x)}

\\=

\frac{p_{data} (1 - D(x)) - p_g D(x)}{D(x)(1 - D(x))}
$$
当且仅当分子 $p_{data} (1 - D(x)) - p_g D(x) = 0$ 是有极值，即：
$$
p_{data} - p_{data}D(x) - p_g D(x) = 0

\\

p_{data} = (p_{data} + p_g) D(x)

\\

D(x) = \frac{p_{data}}{p_{data} + p_g}
$$
GAN的目标是 $p_g = p_{data}$ ，因此 $D(x)$ 的极值是 $1/2$ 。即，当模型训练充分时，判别器的输出期望应该时 0.5 。

---

## 3.3 Score Distillation Loss

蒸馏损失为 ：
$$
L_{distill} (\hat{x}_\theta (x_s, s), \psi) 
\\=
E_{t, \epsilon'} [c(t) d(\hat{x}_\theta, \hat{x}_\psi (sg(\hat{x}_\theta, t); t))]
$$
其中：

- $sg$ 表示 stop-gradient 操作，即教师模型的输出进行梯度截断。
- 直观上，蒸馏损失使用距离度量 $d$ 来度量学生模型的生成图像 $\hat{x}_\theta$ 和教师模型的输出 $\hat{x}_\psi (\hat{x}_{\theta, t}, t) = (\hat{x}_{\theta, t} - \sigma_t \hat{\epsilon}_\psi (\hat{x}_{\theta, t}, t)) / \alpha_t$ 差异。
- 定义距离函数 $d(x, y) := ||x - y||_2^2$ 
- $c(t)$ 是权重函数，有两种函数：
  - 指数权重，$c(t) = \alpha_t$ ，即噪声等级高时，权重系数更小。
  - SDS加权（Score distillation sampling）
  - 后面会证明，使用 $d(x, y) = ||x - y||_2^2$ 时，并选择一个特殊的 $c(t)$ ，上面的蒸馏损失等价于 SDS损失 $L_{SDS}$ 
- 最后，也验证了 noise-free score distillation (NFSD)目标函数，一种最近提出的 SDS 变种

# 4 实验

- 训练了两个模型 ADD-M (860M参数量)和 ADD-XL (3.1B参数量) ：

  - ADD-M 使用SD2.1 backbone，消融实验使用SD1.5对比其他方法

  - ADD-XL使用SDXL backbone

- 所有实验都使用 $512 \times 512$ 分辨率，SDXL输出分辨率高于512的会做下采样。

- 蒸馏权重 $\lambda = 2.5$ 
- R1 penaty 强度 $\gamma = 10^{-5}$ 
- 判别器的条件：
  - 使用 CLIP-ViT-g-14 text encoder来计算 $c_{text}$ 
  - 基于ViT-L的DINOv2的 CLS编码作为图像 embedding $c_{img}$ 

## 4.1 判别器特征提取网络

![image-20231211151302325](./imgs/37-SDXL_Turbo/image-20231211151302325.png)

- DINOv2的效果最好

## 4.2 判别器的condition

![image-20231211151353069](./imgs/37-SDXL_Turbo/image-20231211151353069.png)

- 和StyleGAN-T的结论相同，加入$c_{text}$ 后，效果有提升。
- 此外，$c_{image}$ 的condition比$c_{text}$ 的作用更大。
- 同时结合 $c_{text}$ 和 $c_{image}$ 的效果最好。

## 4.3 学生模型预训练

![image-20231211151553706](./imgs/37-SDXL_Turbo/image-20231211151553706.png)

- 使用预训练的学生模型，效果差距十分明显

## 4.4 损失项

![image-20231211151746484](./imgs/37-SDXL_Turbo/image-20231211151746484.png)

- 对抗损失和蒸馏损失都是必须的。
- 单独使用蒸馏损失的效果不大，但是结合使用两个损失后有明显的提升。
- 蒸馏损失不同的加权策略有较大的差异， 指数策略能够产生更多样的样本，因为 FID 更低
- SDS和NFSD策略能够改善图像质量及文本对齐，如 CS 指标所示。
- 消融实验中使用 指数策略，最终的模型使用NFSD加权策略

## 4.5 教师模型类型

![image-20231211152119275](./imgs/37-SDXL_Turbo/image-20231211152119275.png)

- 有趣的是，更大的教师模型和更大的学生模型都不一定能够产生更好的 FID和CS结果。

## 4.6 教师模型时间步

![image-20231211152432368](./imgs/37-SDXL_Turbo/image-20231211152432368.png)

- 虽然蒸馏损失公式允许教师模型通过采取几个连续的步骤，但我们发现几个步骤并不能最终带来更好的性能。

# 5 附录

## 5.1 SDS作为蒸馏损失的一个特例

![image-20231211153049240](./imgs/37-SDXL_Turbo/image-20231211153049240.png)

- 前提条件是设置 $c(t) = \frac{\alpha_t}{2 \sigma_t} w(t)$ 

- 蒸馏损失中，$\theta$ 只和 $\hat{x}_\theta$ 相关，因此右侧有 $\frac{d \hat{x}_\theta}{d \theta}$ 

- 其中, $w(t)$ 来自于 DreamFusion https://github.com/ashawkey/stable-dreamfusion

  ![image-20231211161920370](./imgs/37-SDXL_Turbo/image-20231211161920370.png)

- 结合消融实验来看，该项的作用并不是很大，因此可以直接使用指数加权。
