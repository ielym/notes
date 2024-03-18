# PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS

**NOTE：**

- 渐进式蒸馏针对的是 VDM，不是 Latent Diffusion Models，所以没有隐变量



# 0 引用

- 论文：https://arxiv.org/pdf/2202.00512.pdf

# 1 介绍

为了解决扩散模型生成一张高质量的图需要推理成百上千个step的问题，本文的主要贡献：

- 给出了一种在少量推理时间步下提升稳定性的方法
- 提出了一种渐进式蒸馏的方法，能够把需要 $N$ 步推理的模型蒸馏至 $N/2$个时间步，重复该操作，最终蒸馏到了4步。

# 2 方法

![image-20231213192436082](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231213192436082.png)

**渐进式蒸馏：**用一个教师模型，蒸馏出一个推理步长减半的学生模型，并重复该过程。

- 先用普通的训练方法训练出一个教师模型。
- 之后从训练集中采样数据进行蒸馏。
- 整理时教师模型从时间步 $t$ 开始，推理2个DDIM step；学生模型也从时间步 $t$ 开始，推理1个 DDIM step。最终学生模型的target是教师模型两步的推理结果。

## 2.1 符号说明

- $N$ ：学生模型的推理步数
- $x$ ：原始图像
- $z_t$ ：原始图像在时间步 $t$ 时刻的加噪图像
- $\tilde{x}$ ：学生模型（也指待训练模型）的 gt 

## 2.2 蒸馏算法

![image-20231213193908464](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231213193908464.png)

### 2.2.1 步长解释

- $N$ 是学生采样的时间步的个数，如 512， 256， 128， 64， 32， 16， 8， 4， 2， 1。
- $i$ 是从 $[1,2,3,4,5,...N]$ 中采样出来的一个数字。需要注意，$i$ 并不是时间步 $t$ ，只是 1 - N 的一个数字！！！
- $t = i / N$ ，表示采样出来的 $i$ 在 $N$ 个推理时间步中所占的比例，取值范围是 $(0, 1]$  。需要注意，$t$ 也并不是通常理解的时间步，$t \times 1000$ 才是时间步，而这个 $1000$ 与 $N$ 无关。如，
  - 当蒸馏到 $N = 512$ 时，推理时的起始时间步是 $i / N \times 1000 = 512 / 512 \times 1000 = 1000$
  - 当蒸馏到 $N = 256$ 时，推理时的起始时间步还是 $i / N \times 1000 = 256 / 256 \times 1000 = 1000$ 
- 这里之所以用 $t = i / N$ ，之后还要乘训练步长，用这么麻烦的时间步计算方式，主要好处是在计算教师步长时比较方便。
- 教师步长计算：
  - 需要明确：
    - 无论 $N$ 是多少，训练都是从 $(1, 1000]$ 中挑时间步（等间隔 for DDIM）。而不是说当 $N=50$ 时，时间步 $T$ 就是 1 到 50，$T$ 还是1 到 1000的，比如1, 50, 100, 150, 200, ...., 950 。 
    -  学生模型和教师模型共用同一个 $z_t$ ，即从同一个时间步加噪的噪声图像开始去噪。
    - 对于 $z_t$ ，教师模型需要跑两遍 UNet去噪；学生模型跑一遍 UNet 去噪，并对齐教师模型去噪两遍的结果。
  - 理解 $t - 0.5/N, t - 1 / N$ ：
    - 从学生模型的角度考虑，在推理时，初始时间步是 $N / N \times 1000$ ；初始时间步和第二个时间步之间的时间步之差是 $1 / N$ ，所以下一次的时间步就是 $(N/N - 1/N) \times 1000$ ，第三次推理的时间步是 $(N/N - 1/N - 1/N) \times 1000$ 。
    - 依次类推，如果当前时间步是 $t$ （没有乘1000），则下一次的推理时间步就是 $t - 1 / N$ 。
    - 需要注意，刚才说的这些 “下一次的时间步” 都指的是学生模型下一次的时间步。而学生模型的下一次，等价于教师模型来说就是下两次，所以教师模型实际上在 $t$ 和 $t - 1 / n$ 中间还有一个时间步。如果采样步长是线性的，那中间的这个时间步自然就是 $t - 0.5/N$ 。 

### 2.2.2 算法梳理

算法很好理解：

- 从学生模型的时间步中采样出一个 $t$ ，并根据 $t$ 再采样出教师模型的两个步长 $t' = t - 0.5 / N$ ，$t'' = t - 1 /N$ 
- 根据时间步 $t$ 对原图 $x$ 加噪，得到 $z_t = \alpha_t x + \sigma_t \epsilon$ 
- 教师模型推理：
  - 推理一次，得到 $t'$ 时刻的加噪图像 $z_{t'}$ 
  - 接着 $z_{t'}$ 再推理一次，得到 $z_{t''}$ 
  - 理解这个过程就行，不用看算法的公式。算法公式的含义：$\hat{X}_\eta (z_t)$ 表示预测的假 $x_0$ ，又因为 $x_t = \alpha_t x_0 + \sigma_t \epsilon$ ，所以 $\epsilon = \frac{x_t - \alpha x_0}{\sigma_t}$ ，所以预测的 $z_{t-1} = \alpha_t' \hat{X}_\eta (z_t) + \sigma_{t'}  \frac{x_t - \alpha x_0}{\sigma_t} = ...$

- 信噪比和 $w(\cdot)$ 暂时不用理解，就当成是为了提升步数蒸馏稳定性而使用的一种损失加权策略。
- 最后对学生模型单步预测的结果 $\hat{X}_\theta (z_t)$ 和教师模型两步预测的结果 $\tilde{x}$ 算损失即可。论文算法中间 $\tilde{x} = \frac{...}{...}$ 那一行表示（不用关注）
  - 学生模型单步去噪的理想结果是 $z_{sssssssssss} = \alpha_{t''} \tilde{x} + \sigma_{t''}  \frac{x_t - \alpha_{t''} \tilde{x}}{\sigma_{t''}} = ...$ 。（学生模型的时间一个去噪时间步就到 $t''$ 了）
  - 理想情况喜爱 $z_{ssssssssssss} = z_{t''}$ ，两式联立，把 $\tilde{x}$ 求出来，就是学生模型的预测目标。
  - 学生模型实际预测的去噪输出是 $\hat{x}_\theta (z_t)$ 。
- 此外，需要注意损失之前的 $w(\lambda_t)$ ，根据作者的描述，该参数在控制步数蒸馏时有较大作用。

# 3 训练损失

- 按照该论文的符号定义和通常的符号定义，有 $x_t = \alpha_t x_0 + \sigma_t \epsilon = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$  ，所以 $\alpha_t^2 + \sigma_t^2 = 1$ 。

- 本文使用的 $\alpha_t$ 的策略是 $\alpha_t = cos(0.5 \pi t)$ ，也是单调递减的，函数图像如下图所示：

  ```python
  import math
  from matplotlib import pyplot
  
  N = 256
  
  ret = []
  for i in range(1, N + 1):
      ts = i / N
      alpha_t = math.cos(0.5 * math.pi * ts)
      print(alpha_t)
      ret.append(alpha_t)
  
  pyplot.plot(range(len(ret)), ret)
  pyplot.show()
  ```

  

  ![image-20231214152112063](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214152112063.png)

  

---

通常网络通过预测真实加噪的噪声 $\epsilon$ 和网络预测的噪声 $\hat{\epsilon}(z_t)$ 来计算损失：
$$
L_\theta = || \epsilon - \hat{\epsilon}(z_t) ||_2^2 = || \frac{1}{\sigma_t} (z_t - \alpha_t x) - \frac{1}{\sigma_t} (z_t - \alpha_t \hat{x}_\theta (z_t)) ||_2^2 = \frac{\alpha_t^2}{\sigma_t^2} || x - \hat{x}_\theta (z_t) ||_2^2
$$

- 可以看出，直接计算噪声的损失，等价于在 $x$ 空间计算损失，并加上一个权重 $w(\lambda_t) = exp(\lambda_t)$ 
- 上式中，$\lambda_t = log(\alpha_t^2 / \sigma_t^2)$ 

---

正常训练中，用上述损失是没问题的，但是在步数蒸馏中，上述损失存在问题，具体分析如下：

- $\alpha_t^2 / \sigma_t^2$ 随时间步的变化如下图所示，可以看出，早期的信噪比特别大，然后剧烈下降：

![image-20231214153851772](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214153851772.png)

- 为了便于观察分析，对信噪比加一个 log：

![image-20231214154000531](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214154000531.png)

- 上图是1000步的信噪比变化曲线，再看看随着学生推理步数 $N$ 减少时信噪比的变化：

![image-20231214154149442](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214154149442.png)

- 可以看到，当蒸馏到8步时，信噪比会越来越接近0（图中的负数是因为加了 log）。

根据上面的分析：

- 由于随着步数蒸馏的进行，信噪比越来越小。由于，$SNR = \alpha_t^2 / \sigma_t^2$ ，意味着 $\alpha_t \to 0$ 。
- 由于 $\hat{x}_\theta (z_t) = \frac{1}{\alpha_t} (z_t - \sigma_t \hat{\epsilon}_\theta (z_t))$ ，所以即使当网络预测的噪声 $\hat{\epsilon}_\theta (z_t)$ 有非常微小的变化，最终预测的原图由于有个系数 $1 / \alpha_t$ ，也会产生较大的变化。
- 这个问题在多步预测时不会对最终的结果造成较大影响，因为随着预测的进行，产生的错误会被逐渐修正，同时也会有 clip 操作。但是减少推理步数时，这个问题产生的影响就很难被修正了。
- 特别是，如果只蒸馏到单步，模型的输入就完全是噪声了，此时信噪比为0，即 $\alpha_t = 0, \sigma_t = 1$ 。此时，预测的噪声和算出的去噪图像之间的关系会因为微小的变化而产生巨大波动，很难建模一个确定的映射关系。

因此，对于步数蒸馏任务，我们需要一种方法，使得最终生成的图像在 $\lambda_t = log[\alpha_t^2 / \sigma_t^2]$ 变化时能够保持稳定，尝试了下列方法，并且发现每种方法都能很好的起作用：

- 直接预测 $x$ 
- 即预测 $x$ 也预测 $\epsilon$ ，两个预测值都在网络的输出通道中，进行split通道获取两个预测值。
- 预测 $v = \alpha_t \epsilon - \sigma_t x$ 

用上面的三种方式直接训练原始非蒸馏的网络，发现也能work well。

除了确定预测方式之外，还需要确定损失的加权系数 $w(\lambda_t)$ ，设计了两种方法：

- `truncated SNR weighting` ： $L_\theta = max(|| x - \hat{x}_t ||_2^2, || \epsilon - \hat{\epsilon}_t ||_2^2) = max(\alpha_t^2 / \sigma_t^2, 1) || x - \hat{x}_t ||_2^2$   
- `SNR + 1 weighting` ：$L_\theta = || v_t - \hat{v}_t ||_2^2 = (1 + \alpha_t^2 / \sigma_t^2) || x - \hat{x}_t ||_2^2$

实验发现，两种方法都能很好的工作。



# 4 预测速度V

![image-20231214161710676](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214161710676.png)

根据文中的符号定义，$x_t = \alpha_t x_0 + \sigma_t \epsilon$ ，因此，已 $x_t$ 可以看作是 $\alpha$ 和 $\sigma$ 的线性组合，可以得到：
$$
tan \phi_t = \frac{\sigma_t}{\alpha_t}
$$

$$
1 = \sqrt{\alpha_t^2 + \sigma_t^2}
$$

因此：

- $\alpha_{\phi} = cos(\phi)$
- $\sigma_{\phi} = sin(\phi)$ 
- 图中的 $z_{\phi} = cos(\phi)x + sin(\phi)\epsilon$ ，这里是向量相加
- 此时 $z_{\phi}, \alpha_{\phi}, \sigma_{\phi}$ 都是和 $\phi$ 相关的随机变量。

定义 $z_{\phi}$ 的速度 $v$ 如上图所示：
$$
v_{\phi} = \frac{d z_{\phi}}{d \phi} = -sin(\phi)x + cos(\phi) \epsilon
$$
因此：
$$
sin(\phi) x = cos(\phi) \epsilon - v_{\phi} 
\\=
\frac{cos(\phi)}{sin(\phi)} (z - cos(\phi)x) - v_{\phi}
$$

$$
sin^2 (\phi) x = cos(\phi) z - cos^2(\phi) x - sin(\phi)v_{\phi}
$$

$$
(sin^2(\phi) + cos^2(\phi))x \\= x \\= cos(\phi) z - sin(\phi) v_{\phi}
$$

用类似的方法，也可以得到：
$$
\epsilon = sin(\phi) z_{\phi} + cos(\phi) v_{\phi}
$$

---

使用 $v$ 作为损失时：

- GT : $v = cos(\phi) \epsilon - sin(\phi) x$
- Pred ：$\hat{v}_{\theta} (z_{\phi}) = cos(\phi) \hat{\epsilon}_\theta(z_{\phi}) - sin(\phi) \hat{x}_{\theta}(z_{\phi})$

---

在 DDIM 中，前一时刻的 $\phi$ 定义成 $\phi_s$ ，因此前一时刻的 $z_{\phi_s}$ 完全按定义为：
$$
z_{\phi_s} = cos(\phi_s) \hat{x}_\theta (z_{\phi_t}) + sin(\phi_s) \hat{\epsilon}_\theta (z_{\phi_t})
$$
![image-20231214180229517](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214180229517.png)

合并：

![image-20231214180302173](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214180302173.png)

最终得到：

![image-20231214180317179](./imgs/40-PROGRESSIVE DISTILLATION FOR FAST SAMPLING OF DIFFUSION MODELS/image-20231214180317179.png)
