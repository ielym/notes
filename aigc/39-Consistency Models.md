# Consistency Models

# 0 引用

- 论文：https://arxiv.org/abs/2303.01469

# 1 介绍

提出一致性模型：

- 支持单步图像生成
- 同时也允许多步生成

一致性模型有两种训练方法：

- 蒸馏
- 直接训，单是也是从预训练模型开始训
- 比现有的步数蒸馏的方法好。

![image-20231220203557692](./imgs/39-Consistency Models/image-20231220203557692.png)

- 基于PF-ODE构建
- 学习一个模型，实现把任意一个step的图像映射到相同的起始点。

# 2 Diffusion Models

一致性模型受连续时间的扩散模型启发。定义 $p_{data}(x)$ 表示真实数据分布，则连续时间扩散模型的 SDE 方程为：
$$
dx_t = f(x_t, t) dt + g(x) dw_t
$$
其中，$t \in [0, T]$ ，$f(\cdot, \cdot)$ 和 $g(\cdot)$ 分别是漂移系数和扩散系数（实值函数），$\{ x_t \}_{t \in [0, T]}$ 表示标准布朗运动。我们定义 $p_0 (x) = p_{data}(x)$ 。在 SDE 中，扩散模型通常定义 $p_T(x)$ 是一个高斯分布 $\pi(x)$ 。

SDE可以表示成ODE的形式：
$$
dx_t = [f(x,t) - \frac{1}{2} g(t)^2 \nabla_x logp_t(x)] dt
$$
其中，$\nabla_x logp_t(x)$ 是 $p_t(x)$ 的score function，因此扩散模型也可以看成是 score-based生成模型。

---

忘记之前的符号。。。。。一致性模型中，把一致性函数定义成了 $f(...)$ ，所以把 SDE，ODE中的 $f()$ 替换成 $\mu()$ ，把 $g()$ 替换成 $\sigma()$ 。。。。。。。。：
$$
dx_t = \mu(x_t, t) dt + \sigma(t) d w_t
$$

$$
dx_t = [\mu(x_t,t) - \frac{1}{2} \sigma(t)^2 \nabla_x logp_t(x_t)] dt
$$

- 通常$p_T(x)$ 接近纯高斯分布 $\pi(x)$ 

- 之后采用 [Elucidating the Design Space of Diffusion-Based Generative Models] 的设置，$\mu(x, t) = 0, \sigma(t) = \sqrt{2t}$ 

- 采样时，首先训练一个分数估计模型 $s_\phi(x, t) \approx \nabla logp_t(x)$ ，然后通过PF-ODE可以得到 $x_t$ 的估计：
  $$
  \frac{dx_t}{dt} = -t s_\phi(x_t, t)
  $$

  - 之后，从 $\hat{x}_T \in \pi = N(0, T^2I)$ 中初始化噪声，并使用ODE求解器如Euler或Heun来获得轨迹 $\{ \hat{x}_t \}_{t \in [0, T]}$ 
  - 为了避免数值不稳定，通常求解器在 $t = \epsilon$ 时就可以停止了，根据 [Elucidating the Design Space of Diffusion-Based Generative Models] 的设置，我们把图像像素所放到 $[-1, 1]$ ，并设置 $T = 80, \epsilon = 0.002$ 。

# 3 Consistency Models

## 3.1 定义

- PF-ODE 的解的轨迹定义成 $\{ x_t \}_{t \in [\epsilon, T]}$ 
- 定义一致性函数 $f : (x_t, t) \to x_{\epsilon}$  。一致性函数有 self-consistency 的属性：对于属于同一个 PF-ODE 的轨迹中的任意 $(x_t, t)$ ，都对应相同的输出。比如，对于任意的 $t, t' \in [t_{min}, t_{max}]$  ，都有 $f(x_t, t) = f(x_{t'}, t')$  

![image-20231221201831396](./imgs/39-Consistency Models/image-20231221201831396.png)

如上图所示，一致性模型用符号 $f_\theta$ 表示，一致性模型的目标是从数据中通过强制学习 self-consitency 的目标来学习一致性函数 $f$ 。

## 3.2 参数化

有两种方式来参数化的表示一致性模型：

- 假设我们有一个深度神经网络 $F_\theta(x, t)$ ，输出维度和 $x$ 相同，则第一种方法可以把一致性模型表示成：

  ![image-20231225150239577](./imgs/39-Consistency Models/image-20231225150239577.png)

- 第二种方法表示一致性模型可以用skip connection来表示：

  ![image-20231225150318486](./imgs/39-Consistency Models/image-20231225150318486.png)

  - $c_{skip} , c_{out}$ 是可微函数。当 $t = \epsilon$ 时，$c_{skip} (\epsilon) = 1$ , $c_{out}(\epsilon) = 0$ 。

## 3.3 采样

当有一个训练好的一致性模型 $f_\theta(\cdot, \cdot)$ 之后，可以初始化噪声 $\hat{x}_T \in N(0, T^2I)$ ，并使用一致性模型直接采样样本 $\hat{x}_\epsilon = f_\theta(\hat{x}_T, T)$ 。此外，也可以基于一致性模型进行多步采样来改善采样质量。

多步采样算法如下图所示：

![image-20231225151653397](./imgs/39-Consistency Models/image-20231225151653397.png)

- 初始化噪声图像 $\hat{x}_T$ 

- 经过一致性模型得到单步采样结果 $x$ 

- 采样出剩余 $N - 1$ 步的时间步 $\tau_1 > \tau_2 > ... > \tau_{N-1}$ ，按照前文所述，对应时间步的图像分布为 $N(0, \tau_n^2I)$ 。

  - 因此，通过前一次的采样结果 $x$ 再重新加噪：
    $$
    \hat{x}_{\tau_n} \in N(0, \tau_n^2I) = N(0, \tau_n^2 - \epsilon^2 + \epsilon^2) = x + \sqrt{\tau_n^2 - \epsilon^2}z
    $$

- 在首次得到 $x$ 之后，重复该过程 $N - 1$ 次，得到最终的输出。

# 4 Consistency Distillation (CD)

- 需要一个预训练好的分数模型 $s_\phi(x, t)$ 。

- 使用 $\frac{dx_t}{dt} = -t s_\phi(x_t, t)$ 作为 PF-ODE方程。

- 考虑离散的时间线 $[\epsilon, T]$ ，把其划分成 $N - 1$ 个子区间 $t_1 = \epsilon < t_2 < ... < t_N = T$ 。

- 其中，根据离散时间步计算连续时间 $t_i$ 的公式使用了 [Elucidating the Design Space of Diffusion-Based Generative Models] 的方法：
  $$
  t_i = ( \epsilon^{1/ \rho} + \frac{i-1}{N-1} (T^{1/ \rho} - \epsilon^{1 / \rho})  ) ^ \rho
  $$
  论文中设置 $\rho = 7$ 函数图像如下图所示：

![image-20231225153855882](./imgs/39-Consistency Models/image-20231225153855882.png)

根据 ODE 方程 $dx_t = [\mu(x_t,t) - \frac{1}{2} \sigma(t)^2 \nabla_x logp_t(x_t)] dt = \Phi(x_t, t) dt$  ，其中 $\Phi(...)$ 表示ODE求解其，如  Euler 求解器，因此：
$$
\frac{dx_t}{dt} = \Phi(x_t, t)
$$
因此，从 $t_{n+1}$ 到 $t_n$ （$dt$ 时间内）的 $x$ 的位移 $d_{x_t}$ 为：
$$
x_{t_{n+1}} - x_{t_n} = d_{x_t} = \frac{dx_t}{dt} dt = \frac{dx_t}{dt} (t_{n+1} - t_n) = (t_{n+1} - t_n)\Phi(x_{t_{n+1}}, t_{n+1}; \phi)
$$
 其中，$\phi$ 表示分数预测网络的参数。因此，$x_{t_n}$ 为：
$$
x_{t_n} =  x_{t_{n+1}} + (t_n - t_{n+1})\Phi(x_{t_{n+1}}, t_{n+1}; \phi)
$$


当使用Euler求解器时，$\Phi(x, t; \phi) = -t s_\phi(x, t)$ ，即：
$$
x_{t_n} =  
x_{t_{n+1}} + (t_n - t_{n+1}) \Phi(x_{t_{n+1}}, t_{n+1}; \phi)
\\=
x_{t_{n+1}} - (t_n - t_{n+1}) t_{n+1} s_\phi(x_{t_{n+1}}, t_{n+1})
$$

---

通过蒸馏训练一致性模型的基本流程如下：

- 采样出 $x \in p_{data}$ 

- 对 $x$ 加噪得到 $N(x, t_{n+1}^2 I)$ 

- 通过ODE求解器（如 Euler），并用预训练好的分数预测网络预测分数，根据下式得到 $\hat{x}_{t_n}$ 
  $$
  \hat{x}_{t_n} = x_{t_{n+1}} - (t_n - t_{n+1}) t_{n+1} s_\phi(x_{t_{n+1}}, t_{n+1})
  $$

- $x_{t_n}$ 和 $x_{t_{n+1}}$ 应具有一致性，即通过一致性模型的输出应该像等：
  $$
  f_\theta(x_{t_{n+1}}, t_{n+1}) = f_{\theta ^ -}(\hat{x}_{t_{n}}, t_{n})
  $$

  - 其中 $f_{\theta ^ -} $ 是 EMA 模型， why ?

- 一致性蒸馏损失定义如下：
  $$
  L_{CD}^N (\theta, \theta^-;\phi) := \mathbb{E} [\lambda(t_n) d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta ^ -}(\hat{x}_{t_{n}}, t_{n}))]
  $$

  - $n \in U[1, N-1]$ 是均匀分布采样的离散时间 $\{ 1, 2, ..., N-1 \}$
  - $x_{t_{n+1}} \in N(x; t_{n+1}^2I)$ 
  - $\lambda(\cdot) \in \mathbb{R}^+$ 是正实数权重
  - $\hat{x}_{t_{n}}$  通过 $\hat{x}_{t_n} = x_{t_{n+1}} - (t_n - t_{n+1}) t_{n+1} s_\phi(x_{t_{n+1}}, t_{n+1})$ 计算得到。需要注意，$\hat{x}_{t_{n}}$ 之所以不用 $N(x; t_{n}^2I)$ 直接采样，是因为一致性模型仅针对同一条轨迹上的点。
  - $d(\cdot, \cdot)$ 是衡量一致性的度量函数，对于任意 $x, y$ ，$d(x, y) \ge 0$ 。并且当且仅当  $x = y$ 时 $d(x, y) = 0$ 
  - 实验发现 $\lambda (t_n) = 1$ 在所有任务和所有数据集上的效果最好。
  - $\theta ^ -$ 表示 EMA 模型的参数，$\theta ^ - = stopgrad(\mu \theta ^- + (1 - \mu) \theta)  $ ，$0 \le \mu \lt 1$  。 ----- 实验发现使用 EMA 比直接用 $\theta$ 能够显著提升训练过程的稳定性，并能够改善最终一致性模型的效果。

 训练算法如下图所示：

![image-20231225190118471](./imgs/39-Consistency Models/image-20231225190118471.png)

# 5 Consistency Training  (CT)

蒸馏训练的方式中，为了计算轨迹上的另一个点 $\hat{x}_{t_n}$ ，需要用到 ODE 采样器，需要预测分数。在没有提前预训练好的分数预测网络时，也可以训练一致性模型，可以用下式代替 $\nabla logp_t(x_t)$ ：
$$
\nabla logp_t(x_t) = -\mathbb{E} [\frac{x_t - x}{t^2}| x_t]
$$
其中，$x \in p_{data}(x)$ ，$x_t \in N(x; t^2 I)$ 。也就是说给定 $x, x_t$ 之后，分数可以直接用 $- (x_t - x) / t^2$ 进行估计。证明如下：

---

根据边缘概率分布：
$$
\nabla logp_t(x_t) = \nabla_{x_t} log \int p_{data}(x) p(x_t | x) dx
$$
其中 $p(x_t | x) = N(x_t; x, t^2 I)$ 。

上式可以进一步简化：
$$
\nabla logp_t(x_t) = \nabla_{x_t} log \int p_{data}(x) p(x_t | x) dx
\\=
\frac{\int p_{data}(x) \nabla_{x_t} p(x_t | x) dx}{\int p_{data}(x) p(x_t | x) dx}   //(log求导)
\\=
\frac{\int p_{data}(x) p(x_t | x) \nabla_{x_t} logp(x_t | x) dx}{\int p_{data}(x) p(x_t | x) dx}
\\=
\frac{\int p_{data}(x) p(x_t | x) \nabla_{x_t} logp(x_t | x) dx}{p_t(x_t)}
\\=
\int \frac{p_{data}(x) p(x_t | x)  }{p_t(x_t)} \nabla_{x_t} logp(x_t | x) dx //(对x积分，x_t是常数)
\\=
\int p(x|x_t) \nabla_{x_t} logp(x_t | x) dx //(贝叶斯定理)
\\=
\mathbb{E}[\nabla_{x_t} logp(x_t | x) | x_t]
\\=
- \mathbb{E}[\frac{x_t - x}{t^2} | x_t] //高斯分布表达式求导
$$

---

因此，$-\mathbb{E} [\frac{x_t - x}{t^2}| x_t]$ 是 $\nabla logp_t(x_t)$ 的无偏估计。

# 6 实验

- 数据：CIFAR-10，ImageNet 64x64，LSUN Bedroom 256x256，LSUN Cat 256x256

## 6.1 距离函数 $d$

对比了：

- $l_2$ 距离： $d(x,y) = || x - y ||_2^2$ 
- $l_1$ 距离：$d(x, y) = || x - y ||_1$ 
- 感知损失：$LPIPS$ 

![image-20240104152820384](./imgs/39-Consistency Models/image-20240104152820384.png)

如上图所示：

- LPIPS显著优于 $l_1, l_2$ ，这是由于训练数据是 CIFAR-10，而 LPIPS 也是为自然图像而涉及的。
- 

## 6.2 ODE求解器

对比了：

- Euler's forward method
- Heun's second order method

![image-20240104153055806](./imgs/39-Consistency Models/image-20240104153055806.png)

- 图b可以看出，在相同的训练时间步N下，Heun's scecond order 求解器都比对应的 Eluer 求解器的效果更好。这是由于在相同的训练时间步N下，使用更高阶的ODE求解器有更小的预测误差。
- 从图b和图c可以看出，Heun + N=18的效果最好

## 6.3 训练CT模型

![image-20240104154545768](./imgs/39-Consistency Models/image-20240104154545768.png)

- 由于CD和CT强相关，因此同样使用了LPIPS
- 如上图所示，更小的训练时间步收敛的更快，但是最终的效果也越差。
