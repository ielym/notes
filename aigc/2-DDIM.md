# Denoising Diffusion Implicit Models

- 论文：https://arxiv.org/abs/2010.02502

# 0 

## 0.1 DDPM

在DDPM中，可以得到：

- $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon$
- $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ 

- 反向过程中，最终的目标是找到 $p(x_{t-1}|x_t)$ ：
  $$
  p(x_{t-1}|x_t) = \frac{p(x_t|x_{t-1}) p(x_{t-1}))}{p(x_t)}
  $$

但是，除了 $p(x_t | x_{t-1})$ 知道之外， $p(x_t), p(x_{t-1})$ 都是未知的。

然而，利用马尔可夫过程的性质，上式可以等价转化为：
$$
p(x_{t-1}|x_t) = p(x_{t-1}|x_t, x_0)  = \frac{p(x_t|x_{t-1}) p(x_{t-1}))}{p(x_t)} = \frac{p(x_t|x_{t-1}, x_0) p(x_{t-1}|x_0))}{p(x_t|x_0)}
$$
最终目标 $p(x_{t-1}|x_t, x_0)$ 中：

- $p(x_t|x_0) = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon} \sim N(\sqrt{\bar{\alpha}_t} x_0, 1 - \bar{\alpha}_t)$ ，$p(x_{t-1}|x_0) = \sqrt{\bar{\alpha}_{t-1}} x_{0} + \sqrt{1 - \bar{\alpha}_{t-1}} \bar{\epsilon} \sim N(\sqrt{\bar{\alpha}_{t-1}} x_0, 1 - \bar{\alpha}_{t-1})$ 
- 唯一未知的 $p(x_t|x_{t-1}, x_0)$ ，根据马尔可夫过程的性质 $p(x_t|x_{t-1}, x_0) = p(x_t|x_{t-1})  = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} z_{t-1} 
  \sim N(\sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)$

- 上述三个分布计算得到：
  $$
  q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  \\ \sim \frac{
  N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_{0} , 1 - \bar{\alpha}_{t-1}) 
  N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
  }
  {N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)}
  
  \\ \propto
  exp
  \{
  -\frac{1}{2}
  
  (
  \frac
  {1 - \bar{\alpha}_{t}}
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  )
  [
  x_{t-1}^2
  -2
  \frac
  {
  
  \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
  + 
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
  
  }
  {1 - \bar{\alpha}_{t}}
  
  x_{t-1}
  ]
  \}
  $$

对照高斯分布的标准形式并化简，可以得到：
$$
q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) 
\sim
N(
x_{t-1};
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}},

\frac
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}
)
$$

- 可以发现，最终的目标服从的分布中：

  - 方差只与 $\alpha_t$ 有关，因此不管是正向过程还是逆向过程，方差一定是个确定的常数

  - 而唯一需要计算的，就只有均值：

    - 正向过程中：
      $$
      \mu_q(x_t, x_0) = 
      \frac
      {
      
      \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
      + 
      \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
      
      }
      {1 - \bar{\alpha}_{t}}
      $$

    - 逆向过程中：
      $$
      \mu_\theta(x_t, t) = 
      \frac
      {
      
      \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
      + 
      \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{x_\theta}(x_t, t)
      
      }
      {1 - \bar{\alpha}_{t}}
      $$

  - 接下来的任务就转化成了计算真实图像 $x_0$ 和预测图像 $\hat{x_\theta}$ 的 L2 距离

- 后面根据计算均值的形式，也可以推导出等价于计算预测噪声的形式，在大一统笔记中有过程。

## 0.2 DDIM

在DDPM中，最终的目标是从一个高斯噪声中去噪，所以需要找到下式的分布：
$$
p(x_{t-1}|x_t) = \frac{p(x_t|x_{t-1}) p(x_{t-1}))}{p(x_t)}
$$
对于右侧未知的项，DDPM可以利用马尔可夫过程的性质，把  $p(x_t|x_{t-1}), p(x_t), p(x_{t-1})$ 等价转换为条件概率 $p(x_t|x_{t-1}, x_0)p(x_t|x_0), p(x_{t-1}|x_0)$ 进行计算。

然而，马尔可夫过程在逆向过程生成时过于耗时，因此DDIM想使用非马尔可夫过程完成前向和逆向推理，就不能再使用上述结论了。

那么，如何获得 $p(x_{t-1} | x_t)$ 实现去噪？DDIM的方法非常直接，既然右侧的三项此时都无法利用马尔可夫的性质了，那就干脆不要右侧三项了，直接求解 $p(x_{t-1} | x_t)$ 

**NOTE : DDIM 的目的是等价替换DDPM的前向/逆向过程，而不是设计一种新方法。**

- 在 DDPM 中：
  $$
  q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) 
  \sim
  N(
  x_{t-1};
  \frac
  {
  
  \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
  + 
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
  
  }
  {1 - \bar{\alpha}_{t}},
  
  \frac
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  {1 - \bar{\alpha}_{t}}
  )
  $$
  是一个高斯分布。等价替换后，在DDIM中，$p(x_{t-1}|x_t)$ 也需要是一个高斯分布，并且也是形如：
  $$
  q(x_{t-1}|x_t) 
  \sim
  N(k x_0 + m x_t, \sigma^2)
  $$
  
- 在DDPM中，依赖于马尔可夫过程，有 $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ ；在 DDIM 中，虽然不依赖马尔可夫过程，也要有 $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ ，即：
  $$
  q(x_{t-1}|x_t) 
  \sim
  N(k x_0 + m (\sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}), \sigma^2)
  $$
  即：
  $$
  x_{t-1} = k x_0 + m (\sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}) + \sigma \epsilon
  \\=
  (k+m\sqrt{\bar{\alpha}_t}) x_0 + m \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon} + \sigma \epsilon
  $$
  后者两个正太分布相加，服从于：
  $$
  m \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon} + \sigma \epsilon \sim N(0, m^2(1 - \bar{\alpha}_t) + \sigma^2)
  $$
  因此：
  $$
  x_{t-1} = k x_0 + m (\sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}) + \sigma \epsilon
  \\=
  (k+m\sqrt{\bar{\alpha}_t}) x_0 + \sqrt{m^2(1 - \bar{\alpha}_t) + \sigma^2} \epsilon
  $$
  为了对齐  $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ ：

  - $k+m\sqrt{\bar{\alpha}_t} = \sqrt{\bar{\alpha}_{t-1}}$
  - $m^2(1 - \bar{\alpha}_t) + \sigma^2 = 1 - \bar{\alpha}_{t-1}$

  因此，可以得到：
  $$
  m = \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}}
  $$

  $$
  k = \sqrt{\bar{\alpha}_{t-1}} - \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}} \sqrt{\bar{\alpha_t}}
  \\=
  \sqrt{\bar{\alpha}_{t-1}} - \sqrt{{1 - \bar{\alpha}_{t-1} - \sigma^2}} \sqrt{\frac{{\bar{\alpha_t}}}{1 - \bar{\alpha}_t}}
  $$

  即，在未使用马尔可夫过程的特性，通过定义另一个分布，并不断对齐DDIM的过程，最终也得到了 $q(x_{t-1}|x_t)$ ，并且这个分布能够保证 $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ 依然成立。
  $$
  q(x_{t-1}|x_t) \sim N(
  (\sqrt{\bar{\alpha}_{t-1}} - \sqrt{{1 - \bar{\alpha}_{t-1} - \sigma^2}} \sqrt{\frac{{\bar{\alpha_t}}}{1 - \bar{\alpha}_t}}) x_0 
  + 
  \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}} x_t, \sigma^2
  )
  
  \\=
  \sqrt{\bar{\alpha}_{t-1}} x_0
  -
  \sqrt{{1 - \bar{\alpha}_{t-1} - \sigma^2}} \sqrt{\frac{{\bar{\alpha_t}}}{1 - \bar{\alpha}_t}} x_0
  +
  \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}} x_t
  
  \\=
  
  \sqrt{\bar{\alpha}_{t-1}} x_0
  -
  \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}} \sqrt{\bar{\alpha}_{t}} x_0
  +
  \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}} x_t
  
  \\=
  
  \sqrt{\bar{\alpha}_{t-1}} x_0
  -
  \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma^2}{1 - \bar{\alpha}_t}} 
  (\sqrt{\bar{\alpha}_{t}} x_0 - x_t)
  
  \\=
  N(
  \sqrt{\bar{\alpha}_{t-1}} x_0
  +
  \sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} 
  \frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2)
  $$

## 0.3 DDIM的采样过程

按照 
$$
q(x_{t-1}|x_t) \sim N(
\sqrt{\bar{\alpha}_{t-1}} x_0
+
\sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2)
$$
进行采样，可以采样得到 $x_{t-1}$：
$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0
+
\sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} 
\frac{x_t - \sqrt{\bar{\alpha}_{t}} x_0}{\sqrt{1 - \bar{\alpha}_t}} + \sigma \epsilon
$$
上式在推导过程中，需要满足DDIM的 $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ ，带入得：
$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} 
(\frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}}{\sqrt{\bar{\alpha}_t}})
+
\sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2} \bar{\epsilon}
+ \sigma \epsilon
$$
由于：

- DDIM使用非马尔可夫过程
- 上述计算中使用了 $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}$ ，该表达式在DDIM中成立，并不是由马尔可夫过程推导出来的，而是人为强制设定的，目标是使得 $q(x_{t-1}|x_t) 
  \sim
  N(k x_0 + m x_t, \sigma^2)$ 中，不使用马尔可夫过程的 $x_t$ 具有和使用马尔可夫过程的 $x_t$ 都能够直接从 $x_0$ 获取到中间结果。

由于DDIM不需要服从马尔可夫的性质，因此上式可以改写为：
$$
x_s = \sqrt{\bar{\alpha}_s} 
(\frac{x_k - \sqrt{1 - \bar{\alpha}_k} \bar{\epsilon}}{\sqrt{\bar{\alpha}_k}})
+
\sqrt{1 - \bar{\alpha}_s - \sigma^2} \bar{\epsilon}
+ \sigma \epsilon
$$
其中， $s \lt k$ 。因此，此时采样过程不是 $t=1,2,3,4$ 的连续值，而是只要满足 $t_1 < t_2 < t_3 < ...$ 即可，即从 $0-T$ 中采样出一个升序的子序列即可。 

对于上式的方差 $\sigma$ ，直接让其等于 DDPM 的方差即可：
$$
\sigma^2 =
\eta 
\frac
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}
$$
推理过程中，与DDPM相同，由网络来预测 $\bar{\epsilon}$ ：
$$
x_s = \sqrt{\bar{\alpha}_s} 
(\frac{x_k - \sqrt{1 - \bar{\alpha}_k} \bar{\epsilon_\theta}(x_t)}{\sqrt{\bar{\alpha}_k}})
+
\sqrt{1 - \bar{\alpha}_s - \sigma^2} \bar{\epsilon_\theta}(x_t)
+ \sigma \epsilon
$$


其中 $\eta \in [0, 1]$ ，

- 当 $\eta=1$ 就是DDPM （后续会推导）
- 当 $\eta = 0$ ，上式中唯一具有随机性的 $\sigma \epsilon = 0$ ，因此采样过程不再具有随机性。给定 $x_T$ ，则 $x_0$ 一定相同。

当 $\eta = 1$ 时，期望为：

![image-20230612223155186](imgs/2-DDIM/image-20230612223155186.png)

![image-20230612223203505](imgs/2-DDIM/image-20230612223203505.png)

#  1 介绍

从去噪自编码器模型中进行采样，通常有两类方法：

- 基于朗之万动力学
- 反转Diffusion的正向过程

这些方法的缺点是需要多轮迭代来产生高质量样本，如DDPM采样5w张32x32的图像需要20小时，而GAN只需要不到一分钟。为了缩小Diffusion模型和GAN的速度差异，本文提出了  Denoising Diffusion Implicit Models (DDIM) 。

# 2 背景

对于数据分布 $q(x_0)$ ，我们的目标是学习一个数据分布 $p_\theta(x_0)$ 来估计 $q(x_0)$ 
