 [DDPM: Denoising Diffusion Probabilistic Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2006.11239) 

扩散模型包括两个过程：前向过程 (forward process) 和 反向过程 (reverse process) ；其中前向过程又称为扩散过程 (diffusion process) 。前向过程和反向过程都是一个参数化的马尔科夫链 (Markov chain)。

- 前向/扩散过程是指对数据逐步增加高斯噪声，直至数据变成随机噪声的过程。

- 反向过程是去噪的过程。即，从一个随机噪声开始，逐步去噪，最终生成一个真实的样本。

![img](imgs/1-diffusion/v2-cbfd418272d00bb3a81518f82ad998d0_720w.webp)

# 1 前向/扩散过程

对于任意时刻 $t$ 的图像 $x_t$ ，显然是由上一时刻的图像 $x_{t-1}$ 和 当前时刻的噪声 $z_{t}$ 叠加产生的：

- $x_{t-1}$ 和 $z_t$ 应该按什么比例叠加呢？即，令 $x_t = a x_{t-1} + b z_t$ ，$a, b$ 分别是多少？ 
- 对于不同时刻，$a$ 和 $b$ 的值都是固定的吗？
- 噪声 $z$ 应该服从什么分布？

在论文中，对于 $x$ 和 $z$ 叠加的比例表示如下：

- 令 $\alpha_t = 1 - \beta_t$ ，其中 $\beta$ 的范围是 $[0.0001, 0.02]$ 。即，$\beta$ 随着 $t$ 的增加而增大。
- 任意 $t$ 时刻的图像为 $X_t = \sqrt{\alpha_t} X_{t-1} + \sqrt{1 - \alpha_t} Z_{t}$
- 由于 $\beta$ 随着 $t$ 的增大而增大，因此上式中 $\alpha$ 逐渐减小，$1 - \alpha$ 逐渐增大。所以越到后期噪声的加权比例也就越大。
- 噪声 $Z \sim N(0, 1)$

从第 $0$ 时刻开始（原图），每一步都按照上述公式添加噪声，进行 $T$ 次之后，就获得了一张几乎完全是随机噪声的图像了。

## 1.1 特性1 ：任意时刻的 $x_t$ 可以由 $x_0$ 和 $\beta$ 表示

但是，如果 $T$ 设置的较大，每一步都重复计算比较冗余。在扩散过程中，一个重要的特性是我们可以直接基于原始数据 $x_0$ 来直接获得任意步 $t$ 的 $x_t$ 图像：

第 $t$ 时刻：
$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} z_{t-1}
$$
其中，$x_{t-1}$ ：
$$
x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} z_{t-2}
$$
其中，$x_{t-2}$ ：
$$
x_{t-2} = \sqrt{\alpha_{t-2}} x_{t-3} + \sqrt{1 - \alpha_{t-2}} z_{t-3}
$$

$$
...
$$

$$
x_{1} = \sqrt{\alpha_{1}} x_{0} + \sqrt{1 - \alpha_{1}} z_{0}
$$

如上，首先把 $x_{t-1}$ 带入到 $x_t$ 的式子中：
$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} z_{t-1}
\\
=
\sqrt{\alpha_t} 
(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} z_{t-2})
+ \sqrt{1 - \alpha_t} z_{t-1}

\\

=

\sqrt{\alpha_t} \sqrt{\alpha_{t-1}} x_{t-2}
 + \sqrt{\alpha_t} \sqrt{1 - \alpha_{t-1}} z_{t-2} 
 + \sqrt{1 - \alpha_t} z_{t-1}
 
\\
 
=
\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}
+
\sqrt{\alpha_t - \alpha_t\alpha_{t-1}} z_{t-2} 
+
\sqrt{1 - \alpha_t} z_{t-1}
$$
其中，由于$z \sim N(\mu=0, \sigma^2 =1)$，因此：

- $E(x) = 0, D(x) = 1$
- 所以：$E(ax) = aE(x) = a\times0 = 0$ ，$D(ax) = a^2D(x) = a^2$ 。（与高斯分布无关）

所以：

- $\sqrt{\alpha_t - \alpha_t\alpha_{t-1}} z_{t-2} \sim N(0, \alpha_t - \alpha_t\alpha_{t-1})$
- $\sqrt{1 - \alpha_t} z_{t-1} \sim N(0, 1 - \alpha_t)$

根据高斯分布的加法性质（$A + B \sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$）：
$$
\sqrt{\alpha_t - \alpha_t\alpha_{t-1}} z_{t-2} 
+
\sqrt{1 - \alpha_t} z_{t-1}
\sim
N(0, \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t)
= N(0, 1 - \alpha_t\alpha_{t-1})
$$
所以，上式中后两项的和可以表示为 $\sqrt{1 - \alpha_t\alpha_{t-1}} \bar{z}_{t-2}$ 。表示从第 $t - 2$ 时刻可以直接采样一个正态分布的噪声 $\bar{z}_{t-2}$ ，直接一步计算得到第 $t$ 时刻的 $x_t$ 。即，上式最终可以表示为：
$$
x_t = 
\sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}} \bar{z}_{t-2}
$$
同理，
$$
x_t = 
\sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}} x_{t-3} + \sqrt{1 - \alpha_t\alpha_{t-1} \alpha_{t-2}} \bar{z}_{t-3}
$$
通过归纳法可得：
$$
x_t = 
\sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1}} x_{0} 
+ 
\sqrt{1 - \alpha_t\alpha_{t-1} \alpha_{t-2}...\alpha_{1}} \bar{z}_{0}

\\
=
\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
其中，$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i = \alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1}$ ，$\bar{z}_0 \sim N(0, 1)$ 。表示，$x_t$ 时刻的图像可以由 $x_0$ 时刻的原始图像和一个标准高斯噪声表示。$\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1 - \bar{\alpha}_t}$ 分别称为 `signal rate` 和 `noise rate`

此外，可以发现，由于 $0 \lt \alpha_t \lt 1$ ，因此当 $t$ 足够大时，$\alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1} = \bar{\alpha}_t$ 就会趋近于0，而 $1 - \bar{\alpha}_t$ 会趋近于1。即，当 $t$ 足够大时，$x_t$ 几乎全部是一个高斯噪声。

# 2 反向过程

反向过程中，我们的初始是第 $t$ 时刻的高斯噪声 $x_t$ ，我们想通过 $x_t$ 来获得 $x_{t-1}$ ，并最终逐步获得 $x_0$ ，同样是一个马尔可夫链。

把从 $x_t$ 获得到的 $x_{t-1}$ 的分布记作 $q(x_{t-1}|x_t)$ ，显然这个分布是未知的。但是在正向过程中，我们很容易获得 $q(x_{t}|x_{t-1})$ 。所以，求解 $q(x_{t-1}|x_t)$ 很自然的就联想到使用贝叶斯公式：
$$
q(A|B) = \frac{q(A)q(B|A)}{q(B)} 
$$
即：
$$
q(x_{t-1}|x_t) = \frac{q(x_{t-1}) q(x_t|x_{t-1})}{q(x_t)}
$$
由于正向过程是一个马尔可夫过程，只与前一时刻的状态有关，因此第 $t$ 时刻的图像是依赖于第 $t-1$ 时刻的。所以，我们在正向过程中只能够获得 $q(x_{t}|x_{t-1})$ ，而 $q(x_t)$ 和 $q(x_{t-1})$ 是未知的：

- 然而，在前向过程中，我们可以根据 $x_0$ 直接获得任意时刻的图像 $x_t$ ，即，$q(x_t|x_0)$ 或 $q(x_{t-1}|x_0)$ 是已知的。

- 为了利用这个已知量，我们需要重新构造一下上述的贝叶斯公式，也就是加上一个 $x_0$ 已知的条件：
  $$
  q(x_{t-1}|x_t,x_0) = \frac{q(x_{t-1} | x_0) q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  $$

- 此时，贝叶斯公式的右侧三项 $q(x_{t-1} | x_0)$ ，$q(x_t|x_{t-1},x_0)$ 和 $q(x_t|x_0)$ 都是已知的了：
  $$
  q(x_{t-1} | x_0) = 
  \sqrt{\bar{\alpha}_{t-1}} x_{0} 
  +
  \sqrt{1 - \bar{\alpha}_{t-1}} \bar{z}_{0}
  
  \sim
  
  N(\sqrt{\bar{\alpha}_{t-1}} x_{0} , 1 - \bar{\alpha}_{t-1})
  $$

  $$
  q(x_t|x_{t-1},x_0) = 
  \sqrt{\alpha_{t}} x_{t-1} 
  +
  \sqrt{1 - \alpha_{t}} z_{t-1}
  
  \sim
  
  N(\sqrt{\alpha_{t}} x_{t-1} , 1 - \alpha_{t})
  $$

  $$
  q(x_t|x_0) = 
  \sqrt{\bar{\alpha}_{t}} x_{0} 
  +
  \sqrt{1 - \bar{\alpha}_{t}} \bar{z}_{0}
  
  \sim
  
  N(\sqrt{\bar{\alpha}_{t}} x_{0}, 1 - \bar{\alpha}_{t})
  $$

- 至此，贝叶斯公式的右侧三项的分布都已知，且都服从高斯分布，自然也能计算出左侧 $q(x_{t-1}|x_t,x_0)$ 的分布：

  - 把右侧三项的均值方差带入到高斯分布的密度函数公式中：
    $$
    q(x_{t-1} | x_0) = 
    
    \frac{1}{\sqrt{2\pi} \sqrt{1 - \bar{\alpha}_{t-1}}} e^{- \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{2 (1 - \bar{\alpha}_{t-1})}}
    $$

    $$
    q(x_t|x_0) = \frac{1}{\sqrt{2\pi} \sqrt{1 - \bar{\alpha}_{t}}} e^{- \frac{(x_t - \sqrt{\bar{\alpha}_{t}} x_{0})^2}{2 (1 - \bar{\alpha}_{t})}}
    $$

    $$
    q(x_t|x_{t-1},x_0) = \frac{1}{\sqrt{2\pi} \sqrt{1 - \alpha_{t-1}}} e^{- \frac{(x_t - \sqrt{\alpha_{t}} x_{t-1})^2}{2 (1 - \alpha_{t-1})}}
    $$

    

- 所以：
  $$
  q(x_{t-1}|x_t,x_0) = \frac{q(x_{t-1} | x_0) q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  
  \\
  = 
  \frac{\frac{1}{\sqrt{2\pi} \sqrt{1 - \bar{\alpha}_{t-1}}} \frac{1}{\sqrt{2\pi} \sqrt{1 - \alpha_{t}}}}{\frac{1}{\sqrt{2\pi} \sqrt{1 - \bar{\alpha}_{t}}}}
  
  Exp
  {(
  
  -\frac{1}{2}
  (\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{(1 - \bar{\alpha}_{t-1})}
  +\frac{(x_t - \sqrt{\alpha_{t}} x_{t-1})^2}{(1 - \alpha_{t})}
  - \frac{(x_t - \sqrt{\bar{\alpha}_{t}} x_{0})^2}{(1 - \bar{\alpha}_{t})})
  )}
  $$

- 由于当 $t$ 确定时，上式系数项为常数，暂时用 $M$ 表示，即：
  $$
  M =
  \frac{\frac{1}{\sqrt{2\pi} \sqrt{1 - \bar{\alpha}_{t-1}}} \frac{1}{\sqrt{2\pi} \sqrt{1 - \alpha_{t}}}}{\frac{1}{\sqrt{2\pi} \sqrt{1 - \bar{\alpha}_{t}}}}
  $$

  $$
  q(x_{t-1}|x_t,x_0) = \frac{q(x_{t-1} | x_0) q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  
  \\
  =
  M
  \cdot
  Exp
  {(
  -\frac{1}{2}
  (\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{(1 - \bar{\alpha}_{t-1})}
  +\frac{(x_t - \sqrt{\alpha_{t}} x_{t-1})^2}{(1 - \alpha_{t})}
  - \frac{(x_t - \sqrt{\bar{\alpha}_{t}} x_{0})^2}{(1 - \bar{\alpha}_{t})})
  )}
  
  \\
  \propto
  
  Exp
  {(
  -\frac{1}{2}
  (\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{(1 - \bar{\alpha}_{t-1})}
  +\frac{(x_t - \sqrt{\alpha_{t}} x_{t-1})^2}{(1 - \alpha_{t})}
  - \frac{(x_t - \sqrt{\bar{\alpha}_{t}} x_{0})^2}{(1 - \bar{\alpha}_{t})})
  )}
  $$

  其中：
  $$
  \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{(1 - \bar{\alpha}_{t-1})}
  +\frac{(x_t - \sqrt{\alpha_{t}} x_{t-1})^2}{(1 - \alpha_{t})}
  - \frac{(x_t - \sqrt{\bar{\alpha}_{t}} x_{0})^2}{(1 - \bar{\alpha}_{t})}
  
  \\
  =
  
  \frac{x_{t-1}^2 + \bar{\alpha}_{t-1}x_0^2 - 2\sqrt{\bar{\alpha}_{t-1}} x_0 x_{t-1}}{(1 - \bar{\alpha}_{t-1})}
  +
  \frac{x_t^2 + \alpha_{t} x_{t-1}^2 - 2 \sqrt{\alpha_{t}} x_t x_{t-1}}{(1 - \alpha_{t})}
  -
  \frac{x_t^2 + \bar{\alpha}_{t} x_0^2 -2\sqrt{\bar{\alpha}_{t}} x_0 x_t}{(1 - \bar{\alpha}_{t})}
  
  \\
  =
  
  \frac{1}{(1 - \bar{\alpha}_{t-1})} x_{t-1}^2 
  - 
  \frac{2\sqrt{\bar{\alpha}_{t-1}} x_0}{(1 - \bar{\alpha}_{t-1})} x_{t-1} 
  +
  \frac{\alpha_t}{(1 - \alpha_{t})} x_{t-1}^2 
  - 
  \frac{2 \sqrt{\alpha_{t}} x_t}{(1 - \alpha_{t})} x_{t-1}
  + C(x_0, x_t)
  
  \\
  =
  [\frac{1}{(1 - \bar{\alpha}_{t-1})} + \frac{\alpha_t}{(1 - \alpha_{t})}] x_{t-1}^2
  -
  [\frac{2\sqrt{\bar{\alpha}_{t-1}} x_0}{(1 - \bar{\alpha}_{t-1})} + \frac{2 \sqrt{\alpha_{t}} x_t}{(1 - \alpha_{t})}] x_{t-1}
  + 
  C(x_0, x_t)
  $$

- 上式中，由于是在 $x_t, x_0$ 已知的条件下计算贝叶斯公式，因此 $x_0, x_t$ 都是已知量。所以把 $x_0$ 和 $x_t$ 相关的项都放在 $C(x_0, x_t)$ 中作为常数项。仅保留未知数 $x_{t-1}$ 相关的项。

- 至此，可以获得：
  $$
  q(x_{t-1}|x_t,x_0) = \frac{q(x_{t-1} | x_0) q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  
  \\
  \propto
  
  Exp(-\frac{1}{2} [[\frac{1}{(1 - \bar{\alpha}_{t-1})} + \frac{\alpha_t}{(1 - \alpha_{t})}] x_{t-1}^2
  -
  [\frac{2\sqrt{\bar{\alpha}_{t-1}} x_0}{(1 - \bar{\alpha}_{t-1})} + \frac{2 \sqrt{\alpha_{t}} x_t}{(1 - \alpha_{t})}] x_{t-1}
  + 
  C(x_0, x_t)])
  $$

- 对照高斯分布的展开形式：
  $$
  Exp(- \frac{(x - \mu)^2}{2\sigma^2}) = Exp(-\frac{1}{2} \frac{x^2 + \mu^2 -2x\mu}{\sigma^2})
  \\=
  Exp(
  -\frac{1}{2} 
  (
  \frac{1}{\sigma^2} x^2
  -
  \frac{2\mu}{\sigma^2} x
  +
  \frac{\mu^2}{\sigma^2}
  )
  )
  $$
  可知：
  $$
  \frac{1}{\sigma^2} = \frac{1}{(1 - \bar{\alpha}_{t-1})} + \frac{\alpha_t}{(1 - \alpha_{t})}
  $$

  $$
  \frac{2\mu}{\sigma^2} = \frac{2\sqrt{\bar{\alpha}_{t-1}} x_0}{(1 - \bar{\alpha}_{t-1})} + \frac{2 \sqrt{\alpha_{t}} x_t}{(1 - \alpha_{t})}
  $$

  $$
  \frac{\mu^2}{\sigma^2} = C(x_0, x_t)
  $$

  即：
  $$
  \sigma^2 = 
  \frac{(1-\alpha_t) (1 - \bar{\alpha}_{t-1})}{1 - \alpha_t \bar{\alpha}_{t-1}}
  = 
  \frac{(1-\alpha_t) (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  $$

  $$
  \mu = \sigma^2 [\frac{\sqrt{\bar{\alpha}_{t-1}} x_0}{(1 - \bar{\alpha}_{t-1})} + \frac{\sqrt{\alpha_{t}} x_t}{(1 - \alpha_{t})}]
  
  \\=
  \frac{(1-\alpha_t) (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  [\frac{\sqrt{\bar{\alpha}_{t-1}}}{(1 - \bar{\alpha}_{t-1})}x_0 + \frac{\sqrt{\alpha_{t}}}{(1 - \alpha_{t})}x_t]
  
  \\=
  \frac{\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}{1 - \bar{\alpha}_{t}} x_0
  +
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} x_t
  $$

  

- 此外，当 $t$ 确定时， $\alpha_t$ ，$\bar{\alpha}_{t-1}$ 都是常数，因此 $\sigma^2$ 是常数。剩下的唯一的变量就是 $\mu$ ，与 $x_t$ 和 $x_0$ 相关。

但是，在反向过程中， $x_t$ 是随机噪声，可以获得，而 $x_0$ 是我们最终想要生成的目标，在计算 $x_{t-1}$ 时，$x_0$ 是无法得到的。

- 但是，正向过程中我们有 $x_t = \sqrt{\bar{\alpha}_t} x_{0} + \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}$ 。反向过程中在获得初始随机噪声 $x_t$ 之后，可以暂时估计出 $x_0$ ：
  $$
  x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
  $$

- 把估计的 $x_0$ 带入，得到：
  $$
  \mu =
  \frac{\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}{1 - \bar{\alpha}_{t}} x_0
  +
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} x_t
  
  \\=
  \frac{\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}{1 - \bar{\alpha}_{t}} 
  \frac{x_t - \sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \bar{z}_{0}
  +
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} x_t
  
  \\=
  
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  x_t
  
  - 
  
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) \sqrt{1 - \bar{\alpha}_t}}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  \bar{z}_{0}
  
  +
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} x_t
  
  
  \\=
  
  (
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  
  + 
  
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  )
  x_t
  
  
  - 
  
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) \sqrt{1 - \bar{\alpha}_t}}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  \bar{z}_{0}
  $$
  其中：
  $$
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) \sqrt{1 - \bar{\alpha}_t}}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  \bar{z}_{0}
  
  \\=
  
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}
  {\sqrt{ \bar{\alpha}_{t}} \sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0}
  
  \\=
  
  \frac
  {\sqrt{\alpha_{t-1} \cdot \alpha_{t-2} \cdot ...} (1-\alpha_t)}
  {\sqrt{ \alpha_{t} \alpha_{t-1} \cdot \alpha_{t-2} \cdot ... } \sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0}
  
  \\=
  
  \frac
  {(1-\alpha_t)}
  {\sqrt{ \alpha_{t}} \sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0}
  $$
  
  $$
  (
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  
  + 
  
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  )
  x_t
  
  \\=
  (
  \frac
  {(1-\alpha_t)}
  {\sqrt{\alpha_{t}}(1 -  \bar{\alpha}_{t})}
  
  + 
  
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  )
  x_t
  
  \\=
  (
  \frac
  {
  1-\alpha_t
  + 
  \alpha_t(1 - \bar{\alpha}_{t-1})
  }
  {\sqrt{\alpha_{t}}(1 -  \bar{\alpha}_{t})}
  
  )
  x_t
  
  \\=
  (
  \frac
  {
  1-\alpha_t
  + 
  \alpha_t - \alpha_t \bar{\alpha}_{t-1}
  }
  {\sqrt{\alpha_{t}}(1 -  \bar{\alpha}_{t})}
  )
  x_t
  
  \\=
  (
  \frac
  {
  1
  -
  \bar{\alpha}_{t}
  }
  {\sqrt{\alpha_{t}}(1 -  \bar{\alpha}_{t})}
  )
  x_t
  
  \\=
  \frac
  {
  1
  }
  {\sqrt{\alpha_{t}}}
  x_t
  $$
  即，化简后的 $\mu$ 为：
  $$
  \mu 
  =
  (
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  
  + 
  
  \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  )
  x_t
  
  
  - 
  
  \frac
  {\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) \sqrt{1 - \bar{\alpha}_t}}
  {\sqrt{ \bar{\alpha}_{t}}(1 -  \bar{\alpha}_{t})}
  \bar{z}_{0}
  
  \\=
  
  -\frac
  {(1-\alpha_t)}
  {\sqrt{ \alpha_{t}} \sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0}
  
  +
  
  
  \frac
  {
  1
  }
  {\sqrt{\alpha_{t}}}
  x_t
  
  \\=
  
  \frac{1}{\sqrt{\alpha_{t}}}
  
  (x_t - \frac
  {(1-\alpha_t)}
  {\sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0})
  $$
  
- 至此，我们可以获得 $q(x_{t-1}|x_t,x_0)$ 的参数：
  $$
  \sigma^2 = 
  \frac{(1-\alpha_t) (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}
  $$

  $$
  \mu 
  =
  \frac{1}{\sqrt{\alpha_{t}}}
  
  (x_t - \frac
  {(1-\alpha_t)}
  {\sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0})
  $$

  - 根据 $\mu, \sigma^2$ ，也就知道了 $x_{t-1}$ 的分布。可以按照该分布采样出任意个 $x_{t-1}$ 
  - $\mu$ 本来是和 $x_0, x_t$ 相关的，但 $x_0$ 就是我们的最终目标，并且在 $t-1$ 时刻该目标还是未知的，因此使用 $\bar{z}_0$ 对 $x_0$ 进行了替换。
    - $\bar{z}_0$ 是一个标准正态分布的噪声，但是在我们假设最终目标 $x_0$ 确定的情况下，$\bar{z}_0$ 也就确定了。因此 $\bar{z}_0$ 是我们假设站在上帝视角，在知道最终的目标图像 $x_0$ 的情况下才能获取得到的。
    - 然而推理过程中，我们并不能站在上帝视角获取到 $x_0$ ，也就无法知道 $\bar{z}_0$ 。
    - 因此，$\bar{z}_0$ 这个标准正态分布的噪声就需要交给神经网络来进行预测。

## 2.1 特性2：重参数技巧

- 为了获得 $x_{t-1}$ ，在上述计算中，通过生成随机的高斯噪声获得了 $x_t$ ，并且网络预测出了 $\bar{z}_0$ ，从而得到了 $x_{t-1}$ 的分布 $q(x_{t-1}|x_t,x_0)$ 

- 按照分布 $q(x_{t-1}|x_t,x_0)$ ，参数 $\mu, \sigma^2$ 已知，就可以按照该分布采样出一个 $x_{t-1}$ 。

- 然而，随机采样的 $x_{t-1}$ ，在后续的反向传播算法中是无法反传梯度的。类似于 gumbel-softmax，Diffusion 这里采用了重参数技巧：
  $$
  x_{t-1} = 
  
  \frac{1}{\sqrt{\alpha_{t}}}
  (x_t - \frac
  {(1-\alpha_t)}
  {\sqrt{1 -  \bar{\alpha}_{t}}}
  \bar{z}_{0})
  
  +
  
  \sigma z
  $$
   其中，$z \sim N(0, 1)$ 是从标准正态分布中采样出来的，这样就把不可导的项转移到了 $z$ 中。而根据高斯分布的性质：$A \sim N(0, 1)$ ，$a + bA \sim N(a, b)$ ，所以上式中的 $x_{t-1}$ 仍然满足均值为 $\mu$ ，方差为 $\sigma$ 。