Understanding Diffusion Models: A Unified Perspective

# 1 Introduction : Generative Models

对于从特定分布中观察到的样本 $x$ ，生成模型的目标是学习如何建模该特定分布 $p(x)$ 。一旦学习到了 $p(x)$ ，我们就可以从该分布中生成新的样本。

生成模型有几个著名的方向：

- Generative Adversarial Networks (GANs) ：通过对抗的方式，建模复杂分布的采样过程
- Likelihood-based ：包含 Autoregressive models, Normalizing Flows, Variational Autoencoders (VAEs)
- Energy-based : 数据分布会被作为任意灵活的能量函数进行学习，之后进行归一化
- Score-based ：基于分数的生成模型是高度相关的;它们不是学习对能量函数本身进行建模，而是将基于能量的模型作为神经网络来学习

本文将从 Likelihood-based 和 score-based 来解释 Diffusion Models

# 2 Background : ELBO, VAE, Hierarchical VAE

我们的最终目标是找到数据的真实分布 $p(x)$ 。直接估计 $p(x)$ 可能是非常复杂的，因为 $x$ 的维度可能非常大。因此，我们想利用已有的观察到的样本 $x$ ，来找到一个更加本质，且比 $p(x)$ 分布更加简单，且一般比 $p(x)$ 更加低维的另外一个分布 $p(z|x)$ 来代替数据的真实分布。

利用贝叶斯公式，$p(z|x)$ 可以表示为：
$$
p(z|x) = \frac{p(z)p(x|z)}{p(x)}
$$
其中：

- $p(z)$, $p(x|z)$ 都是未知的
- $p(x) = \int_z p(x, z) dz$ 可以从联合概率分布的边缘分布计算得到，但是  $p(x, z)$ 也是未知的

然而，神经网络给我们提供了一个很好的黑盒工具箱，我们可以把估计 $p(z|x)$ 的任务交给神经网络来做，让神经网络帮我们估计出 $q_\phi (z|x)$ 。

当然，$q_\phi(z|x)$ 需要与 $p(z|x)$ 足够接近，我们用 KL 散度来衡量两个分布的近似程度：
$$
KL(q_\phi (z|x)|| p(z|x))
\\=
\int_z  q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(z|x)}) dz
\\=
\sum_z q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(z|x)})
$$
上式中，连续的用积分，离散的用累加。如无特殊说明，后面都用积分来表示。

因此，我们的问题是最小化 KL 散度 
$$
KL(q_\phi (z|x)|| p(z|x)) = \int_z  q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(z|x)}) dz
$$


其中， $p(z|x)$ 还是我们最终的目标，还是未知的，我们继续替换：
$$
p(z|x) = \frac{p(x, z)}{p(x)}
$$
即，
$$
KL(q_\phi (z|x)|| p(z|x)) = \int_z  q_\phi (z|x) log(\frac{p(x)q_\phi (z|x)}{p(x,z)}) dz
\\=
\int_z  q_\phi (z|x) [log(p(x)) + log(\frac{q_\phi (z|x)}{p(x,z)})] dz
\\=
\int_z  q_\phi (z|x)log(p(x))dz + \int_z  q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(x,z)})dz
$$
根据期望的定义，$\int_z q(z) f(z, \cdot) dz$ 是 $f(z, \cdot)$ 的期望，因此上式最终可以表示为：
$$
KL(q_\phi (z|x)|| p(z|x)) =  E[log(p(x))] + E[log(\frac{q_\phi(z|x)}{p(x, z)})]
$$
由于 $log(p(x))$ 与我们想要估计的 $q_\phi (z|x)$ 无关，因此可以看作为常数，所以期望等于自身。即，
$$
log(p(x)) = KL(q_\phi (z|x)|| p(z|x)) - E[log(\frac{q_\phi(z|x)}{p(x, z)})]
\\=
KL(q_\phi (z|x)|| p(z|x)) + E[log(\frac{p(x, z)}{q_\phi(z|x)})]
$$
由于 KL 散度 >=0 ，因此
$$
log(p(x)) \le E[log(\frac{p(x, z)}{q_\phi(z|x)})] = ELBO
$$
即，ELBO 是 $log(p(x))$ 的下界。

---

再回过头来梳理一下:

- 我们有了一批观测样本 $x$ ，想要获取样本的真实分布 $p(x)$ 
- 然而，生成图像的真实分布 $p(x)$ 的表达式我们不知道，这个分布可能也是非常复杂的，我们无法直接获取。
- 因此，我们考虑到使用一个隐变量，即找到一个相对简单的中间分布，来表示生成样本的分布 $p(z)$ 
- 为了估计 $p(z)$ ，我们可以使用已经观测到的样本 $x$ 来计算后验概率分布 $p(z|x)$ 
- 然而不幸的是，虽然 $p(z|x)$ 比 $p(x)$ 要简单且维度更低，但是我们也是无法估计出来的。因此我们又想到用神经网络来帮我们估计 $p(z|x)$ ，即 $p_\phi (z|x)$ ，其中 $\phi$ 是神经网络的参数。
- 为了使神经网络估计的 $p_\phi(z|x)$ 与真实的后验分布 $p(z|x)$ 接近，我们选择最小化二者的 KL散度 
- 然而由于根本就不知道真实后验分布 $p(z|x)$ 到底是什么，所以KL散度也无法计算，因此我们又使用 $p(z|x) = \frac{p(x, z)}{p(x)}$ 来代替 $p(z|x)$ 
- 接下来的推到中，我们意外的发现，真实分布$p(x)$的$log$ 形式 $log(p(x)) = KL(q_\phi (z|x)|| p(z|x)) + E[log(\frac{p(x, z)}{q_\phi(z|x)})] \ge ELBO = E[log(\frac{p(x, z)}{q_\phi(z|x)})]$ 

- 再重新思考极大似然估计，最大化 $log(p(x))$ 也就是我们最终的目标，该目标等价于最大化 $ELBO$ ，而 $ELBO$ 中，$q_\phi(z|x)$ 是模型预测出来的，还剩下一个 $p(x, z)$ ，怎么解决这一项，就对应了不同的方法了。

## 2.1 VAE

目标不变，还是最大化 $ELBO = E[log(\frac{p(x, z)}{q_\phi(z|x)})]$ :

- $q_\phi(z|x)$ 是根据输入 $x$ ，模型预测的隐变量 $z$ 的分布

- $p(x, z)$ 未知，但可以等价替换为：
  $$
  p(x, z) = p(z)p(x|z)
  $$

因此，$ELBO$ 可以表示为：
$$
E[log(\frac{p(x, z)}{q_\phi(z|x)})]
\\=
E[log(\frac{p(z)p(x|z)}{q_\phi(z|x)})]
\\=
E[log(p(x|z))] + E[log(\frac{p(z)}{q_\phi(z|x)})]
$$

- 第一项 $log(p(x|z))$ 的期望，就是 decoder 根据隐变量来重建 $x$ 的极大似然。直觉上，好的解码器能够从隐变量 $z$ 上有效的恢复出输入 $x$ 。该项被称为重构项。

- 第二项 :
  $$
  E[log(\frac{p(z)}{q_\phi(z|x)})] = \int_z q_\phi(z|x) log(\frac{p(z)}{q_\phi(z|x)}) dz
  \\=
  - KL(q_\phi(z|x)||p(z))
  $$
   表示编码器从 $x$ 中编码的隐变量 $z$ 和先验分布 $p(z)$ 的相似程度。该项被称为匹配项。

- 因此，我们的目标是最大化第一项（极大似然），并最小化第二项（KL散度等于0表示两个分布相同）。该优化目标等价于最大化 $ELBO$ ：
  $$
  ELBO = E[log(p(x|z))] - KL(q_\phi(z|x)||p(z))
  $$

其中，除了 $q_\phi (z|x)$ 是模型预测的之外，其他项均未知。VAE 的处理如下：

- 既然使用隐变量，而不是去估计原始高维复杂的 $p(x)$ ，就是因为隐变量更加简单，因此干脆假设 $p(z)$ 服从于标准的多元高斯分布，即  $p(z) \sim N(z; 0, 1)$ 。而后验分布实际服从于 $q_\phi(z|x) \sim N(z; \mu, \sigma^2)$ 

- 这样，给定观测图像 $x$ ，我们就可以从 $q_\phi (z|x)$ 中采样出 $z^{(l)}$ ，同时为了考虑 $z^{(l)}$  的可导性，使用重参数法进行采样：
  $$
  z^{(l)} = \mu + \sigma \epsilon, \epsilon \sim N(\epsilon, 0, 1)
  $$

- 因此：
  $$
  argmax_{\phi} ELBO = argmax_{\phi} E[log(p(x|z))] - KL(q_\phi(z|x)||p(z))
  \\=
  argmax_{\phi} E[log(p(x|z^{(l)}))] - KL(q_\phi(z|x)||p(z))
  $$

- 至此，第一项 $E[log(p(x|z^{(l)}))$ 可以使用decoder，并把 encoder 的 $q_\phi (z|x)$ 中重建。第二项在假设了 $p(z) \sim N(z; 0, 1)$ ，以及 $q_\phi(z|x) \sim N(z; \mu, \sigma^2)$ ，所以也可以直接计算解析解。因此整个优化目标也就可以计算了。



## 2.2 HVAE

Hierarchical Variational Autoencoders (HVAE) 假设隐变量有多个层级。如，在 Plato’s cave dweller 中，人们看到墙壁上的倒影，认为隐变量是3维空间的，而3维空间可能也有隐变量，3维空间的隐变量可能也有更高维的隐变量，... 。

这里我们关注一种特殊的 HVAE，即每个隐变量都仅和前一个隐变量相关，即马尔可夫过程，如下图所示（T个隐变量）：

![image-20230528233802489](imgs/2-Understanding%20Diffusion%20Models%20A%20Unified%20Perspective/image-20230528233802489.png)

其中，从观测样本 $x$ 可以获得隐变量 $q(z_1|x)$ ，从 $z_1$ 可以获得隐变量 $q(z_2|x)$ ，... 。反之，从第T层的隐变量 $z_T$ ，我们可以重构隐变量 $p(z_{T-1}|T)$ ，.... 。即：
$$
p(x, z_{1:T}) = p(z_T) p(z_{t-1}|z_T) p(z_{t-2}|z_{T-1}) ... p(x|z_1)
\\=
p(z_T) \prod_{t=2}^{T} p(z_{t-1}|z_{t}) p(x|z_1)
$$
同理：
$$
q(z_{1:T}|x) = q_\phi(z_1|x) \prod_{t=2}^{T} q_\phi(z_t|z_{t-1})
$$
按照 VAE 的计算步骤，我们可以获得：
$$
log(p(x)) \ge ELBO = E[log(\frac{p(x, z_{1:T})}{q_\phi(z_{1:T}|x)})]
\\=
E[log(\frac{p(z_T) \prod_{t=2}^{T} p(z_{t-1}|z_{t}) p(x|z_1)}{q_\phi(z_1|x) \prod_{t=2}^{T} q_\phi(z_t|z_{t-1})})]
$$
其中，$q(z_t)$ 服从什么分布本文没有介绍。但该形式只是为了接下来研究 Diffusion 模型来使用。

# 3 Variational Diffusion Model

从 HVAE 的角度来解释 Variation Diffusion Model (VDM)，最简单的是添加三个约束：

- 隐变量的维度完全等于观测数据的维度
- 每一步生成隐变量的过程是确定的，即在上一步的基础上加上高斯噪声，并且每个step的encoder不是学习来的。
- 最后一步的输出是一个标准高斯噪声

根据上述三个约束：

- **隐变量的维度完全等于观测数据的维度** 。可以统一的表示输入图像 $x_0$ 和隐变量 $x_t$ ，因此，后验分布  $q(x_{1:T}|x_0)$ 可以表示为：
  $$
  q(x_{1:T}|x_0) = q(x_1|x_0) q(x_2|x_1) ... q(x_T|x_{T-1}) 
  \\=
  \prod_{t=1}^{T} q(x_t|x_{t-1})
  $$

- **每一步生成隐变量的过程是确定的，即在上一步的基础上加上高斯噪声，并且每个step的encoder不是学习来的。** 生成每一步的隐变量的过程都是一个固定的线性高斯模型，高斯分布的均值和方差是提前设置的超参数，定义为每个 Gaussian encoder 为：
  $$
  x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} z_{t-1}
  $$
  其中，$z \sim N(z; 0, 1)$ 即：
  $$
  q(x_t|x_{t-1}) \sim N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
  $$

- **最后一步的输出是一个标准高斯噪声** 。即 $p(x_T) \sim N(x_T, 0, 1)$

因此，上述三个假设表示：从输入图像开始，逐步添加高斯噪声，直到图像完全变成高斯噪声。该过程如下所示：

![image-20230529100318573](imgs/2-Understanding%20Diffusion%20Models%20A%20Unified%20Perspective/image-20230529100318573.png)

- 正向过程是添加噪声的过程：
  $$
  q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})
  $$

- 反向过程是去除噪声的过程：
  $$
  p(x_{0:T-1}|x_T) = \prod_{t=1}^{T} p(x_{t-1}|x_t)
  $$

  $$
  p(x_{0:T}) = p(x_{0:T-1}|x_T)p(x_T) = p(x_T) \prod_{t=1}^{T} p(x_{t-1}|x_t)
  $$

在正向过程中，每次添加的都是提前设置好均值和方差的高斯噪声。因此正向过程 （生成隐变量，encoder）没有可学习参数，每一步 $q(x_t|x_{t-1}) \sim N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t) $ 都是已知的。

而反向过程中， 只知道最后一步 $p(x_T) \sim N(0, 1)$ ，而并不知道 $p(x_{t-1}|x_t)$ 的分布，因此需要交给神经网络来估计。即，每一步的 $p(x_{t-1}|x_t)$ 实际上是有可学习参数的，因此上式改写成：
$$
p_\theta(x_{0:T}) = p(x_{0:T-1}|x_T)p(x_T) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)
$$
之后，使用 HVAE 的方式，VDM 可以表示为：
$$
log(p(x)) \ge ELBO = E[log(\frac{p(x, z_{1:T})}{q_\phi(z_{1:T}|x)})]
\\=
E[log(\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)})]
$$
上式中，需要注意 $z， x$ 的维度相同，因此统一都改写成了 $x$ 。
$$
log(p(x)) \ge ELBO = E[log(\frac{p(x, z_{1:T})}{q_\phi(z_{1:T}|x)})]
\\=
E[log(\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)})]
\\=
E[log(\frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T} q(x_t|x_{t-1})})]
$$
由于正向过程和逆向过程的最终目标分别是找到 $q(x_T|x_{T-1})$ $p_\theta (x_0|x_1)$ ，因此先把这两项单独拆出来：
$$
ELBO =
E[log(\frac{p(x_T) p_\theta (x_0|x_1) \prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{q(x_T|x_{T-1}) \prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
\\=
E[log(\frac{p(x_T) p_\theta (x_0|x_1) }{q(x_T|x_{T-1}) )})] 
+
E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
\\=
E[log(p_\theta (x_0|x_1))] 
+
E[log(\frac{p(x_T) }{q(x_T|x_{T-1}) )})] 
+
E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
\\=
E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
+
E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
$$
上式中：

- 第一项 $E[log(p_\theta (x_0|x_1))] $ 为重构项

- 第二项 $E[KL(q(x_T|x_{T-1})||p(x_T))] $ 为先验分布匹配项（$p(x_T) \sim N(0, 1)$）

- 第三项可以继续化简：
  $$
  E[log(\frac{\prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
  \\=
  E[log(\frac{\prod_{t=1}^{T-1} p_\theta(x_{t}|x_{t+1})}{\prod_{t=1}^{T-1} q(x_t|x_{t-1})})]
  \\=
  E[\sum_{t=1}^{T-1} log(\frac{p_\theta(x_{t}|x_{t+1})}{q(x_t|x_{t-1})})]
  \\=
  \sum_{t=1}^{T-1}  E[log(\frac{p_\theta(x_{t}|x_{t+1})}{q(x_t|x_{t-1})})]
  \\=
  -\sum_{t=1}^{T-1} KL({q(x_t|x_{t-1})}||{p_\theta(x_{t}|x_{t+1})})
  $$
  为一致项，表示正向和逆向过程中的 $x_t$ 需要一致。

然而，第三项为了计算 $q(x_t|x_{t-1})$ 和 $p_\theta(x_{t}|x_{t+1})$ ，我们需要对所有的时间步都进行采样，计算量较大，因此，可以根据贝叶斯定理，对正向过程进行简化:
$$
q(x_t|x_{t-1},x_0) = \frac{q(x_{t-1}|x_t, x_0) q(x_t|x_0)}{q(x_{t-1}|x_0)}
$$
上式中，之所以 $q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1})$ 是根据马尔可夫过程：$t$ 时刻只与 $t-1$时刻有关，而与其他时刻无关。

因此，$ELBO$ 可以整体重写为：
$$
ELBO =
E[log(\frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^{T} q(x_t|x_{t-1})})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1) \prod_{t=2}^{T} p_\theta(x_{t-1}|x_t)}{q(x_1|x_0)\prod_{t=2}^{T} q(x_t|x_{t-1})})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) + log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) 
+ 
log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{\frac{q(x_{t-1}|x_t, x_0) q(x_t|x_0)}{q(x_{t-1}|x_0)}})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) 
+
log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})
+
log(\prod_{t=2}^{T}\frac{q(x_{t-1}|x_0) }{q(x_t|x_0)})]
$$
其中最后一项：
$$
log(\prod_{t=2}^{T}\frac{q(x_{t-1}|x_0) }{q(x_t|x_0)})]
\\=
log(
\frac{q(x_{1}|x_0) }{q(x_2|x_0)}
\frac{q(x_{2}|x_0) }{q(x_3|x_0)}
...
\frac{q(x_{T-1}|x_0) }{q(x_T|x_0)}
)
\\=
log(\frac{q(x_{1}|x_0) }{q(x_T|x_0)})]
$$
因此：
$$
ELBO =
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_1|x_0)}) 
+
log(\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})
+
log(\frac{q(x_{1}|x_0) }{q(x_T|x_0)})]
\\=
E[log(\frac{p(x_T) p_\theta(x_0|x_1)}{q(x_T|x_0)}) 
+
\sum_{t=2}^{T} log(\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})]
\\=
E[(log(p_\theta(x_0|x_1)))]
+
E[log(\frac{p(x_T)}{q(x_T|x_0)})]
+
\sum_{t=2}^{T} E[log(\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)})]
\\=
E[(log(p_\theta(x_0|x_1)))]
-
KL(q(x_T|x_0)||p(x_T))
-
\sum_{t=2}^{T} E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]
$$

- 第一项 $E[(log(p_\theta(x_0|x_1)))]$ 是重构项
- 第二项 $KL(q(x_T|x_0)||p(x_T))$ 是先验匹配项，表示最终的噪声需要接近标准高斯分布的噪声
- 第三项 $E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]$ 是去噪匹配项。$p_\theta(x_{t-1}|x_t)$ 表示逆向过程的噪声分布，$q(x_{t-1}|x_t, x_0)$ 表示前向过程的噪声分布。关于为什么这里的噪声匹配项比刚才推导的 $KL({q(x_t|x_{t-1})}||{p_\theta(x_{t}|x_{t+1})})$ 更好，解释如下：
  - **----------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

上式中的三项：

- $E[(log(p_\theta(x_0|x_1)))]$ 是模型预测结果，与 $x_0$ 计算误差即可

- $KL(q(x_T|x_0)||p(x_T))$ 中的 $q(x_T|x_0)$ 是提前设置好的均值和方差，$p(x_T)$ 是标准高斯分布。

- $E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]$ 中，$p_\theta(x_{t-1}|x_t)$ 是模型预测结果，$q(x_{t-1}|x_t, x_0)$ 和  未知：

  由贝叶斯定理和马尔可夫过程可知：
  $$
  q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  $$
  并且，由于不同时刻的隐变量都是线性高斯模型，因此：
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
  
  \sim N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)
  $$
  即，此时可以直接从 $x_0$ 获取任意时刻的隐变量 $x_t$ 。

  此外，从 $x_{t-1}$ 获取 $x_t$ 的方式为 :
  $$
  x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} z_{t-1} 
  \sim N(\sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
  $$
  所以，
  $$
  q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
  \\ \sim \frac{
  N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_{0} , 1 - \bar{\alpha}_{t-1}) 
  N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
  }
  {N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)}
  
  \\ \propto
  exp\{
  -[
  
  \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{2 (1 - \alpha_t)}
  +
  \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{2 (1 - \bar{\alpha}_{t-1})}
  +
  \frac{(x_t - \sqrt{\bar{\alpha}_t} x_{0})^2}{2(1 - \bar{\alpha}_t)}
  ]
  \}
  
  \\=
  
  exp
  \{
  -\frac{1}{2}
  [
  \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{(1 - \alpha_t)}
  +
  \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{(1 - \bar{\alpha}_{t-1})}
  +
  \frac{(x_t - \sqrt{\bar{\alpha}_t} x_{0})^2}{(1 - \bar{\alpha}_t)}
  ]
  \}
  
  \\=
  exp
  \{
  -\frac{1}{2}
  [
  \frac{\alpha_tx_{t-1}^2 - 2\sqrt{\alpha_t}x_tx_{t-1}}{(1 - \alpha_t)}
  +
  \frac{x_{t-1}^2 -2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1}}{(1 - \bar{\alpha}_{t-1})}
  +
  C(x_t, x_0)
  ]
  \}
  
  \\ \propto
  exp
  \{
  -\frac{1}{2}
  [
  \frac{\alpha_tx_{t-1}^2}{(1 - \alpha_t)}
  -
  \frac{2\sqrt{\alpha_t}x_tx_{t-1}}{(1 - \alpha_t)}
  +
  \frac{x_{t-1}^2}{(1 - \bar{\alpha}_{t-1})}
  -
  \frac{2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1}}{(1 - \bar{\alpha}_{t-1})}
  ]
  \}
  
  \\=
  exp
  \{
  -\frac{1}{2}
  [
  (\frac{\alpha_t}{(1 - \alpha_t)} + \frac{1}{(1 - \bar{\alpha}_{t-1})})x_{t-1}^2
  -2
  (\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
  ]
  \}
  
  \\=
  exp
  \{
  -\frac{1}{2}
  [
  (\frac{
  \alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)
  }{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})})x_{t-1}^2
  -2
  (\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
  ]
  \}
  
  \\=
  exp
  \{
  -\frac{1}{2}
  [
  (
  \frac
  {\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  )
  
  x_{t-1}^2
  -2
  (\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
  ]
  \}
  
  \\=
  exp
  \{
  -\frac{1}{2}
  [
  (
  \frac
  {1 - \bar{\alpha}_{t}}
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  )
  
  x_{t-1}^2
  -2
  (\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})x_{t-1}
  ]
  \}
  
  \\=
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
  {(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})})}
  {\frac
  {1 - \bar{\alpha}_{t}}
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}}
  
  x_{t-1}
  ]
  \}
  
  \\=
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
  {(\frac{\sqrt{\alpha_t}x_t}{(1 - \alpha_t)} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{(1 - \bar{\alpha}_{t-1})}) (1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  {1 - \bar{\alpha}_{t}}
  
  x_{t-1}
  ]
  \}
  
  \\=
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
  对照高斯分布的标准形式：
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
  
  \\=
  Exp(
  -\frac{1}{2} 
  (
  \frac{1}{\sigma^2}
  [
   x^2
   -
   2\mu x
  ]
  +C
  )
  $$
  可知：
  $$
  \sigma^2 =
  \frac
  {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
  {1 - \bar{\alpha}_{t}}
  
  \\ 
  \mu =
  \frac
  {
  
  \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
  + 
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
  
  }
  {1 - \bar{\alpha}_{t}}
  $$
  

即：
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
可以发现，$q(x_{t-1}|x_t, x_0)$ 的均值与 $t, x_t, x_0$ 有关，方差只与 $t$ 有关。因此，为了后续表示的简单性，把均值方差分别表示为关于 $x_t, x_0$ 或 $t$ 的函数，即  $\mu_q(x_t, x_0),  \sigma_q(t)$

至此，$ELBO$ 的第三项 $E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]$ 包括：

- $p_\theta(x_{t-1}|x_t))$ ：模型预测的噪声，其均值和方差记作 $\mu_\theta, \sigma_\theta$

- $q(x_{t-1}|x_t, x_0)$ ：噪声的真值，其均值和方差记作 $\mu_q(x_t, x_0), \sigma_q(t)$。

- 接下来，就需要最小化 KL散度 $KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))$  。由于两个分布都是高斯分布，可以直接使用多元高斯分布的KL散度公式，这里略去，可以得到最小化KL散度的最终化简形式为：
  $$
  argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
  \\=
  argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
  $$

  - 其中详细计算过程见论文中公式87 - 公式92

  - 由于噪声的真值 $\sigma_q(t)$ 只和 $t$ 有关，因此论文中这里首先令预测值的方差和真实值的方差相等，之后再进行的计算。**这里如何保证预测方差和真值的方差相同？**

    **-------------------------------------------------------------------------------------------------------------------------------------------**

- $\mu_\theta$ 是模型预测结果的均值，预测过程中，$\mu_\theta$ 只与 $x_T$ 和 $t$ 有关，因此我们也可以把其写成类似于 $\mu_q(x_t, x_0)$ 的形式：
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
  其中，$\hat{x_\theta}(x_t, t)$ 是我们通过噪声输入，来预测出的原始图像。

- 因此，最小化KL散度可以表示为：
  $$
  argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
  \\=
  argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
  \\=
  argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
  \frac
  {
  
  \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
  + 
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{x_\theta}(x_t, t)
  
  }
  {1 - \bar{\alpha}_{t}}
  
  
  -
  
  
  \frac
  {
  
  \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
  + 
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
  
  }
  {1 - \bar{\alpha}_{t}}
  ||^2_2]
  
  \\=
  
  argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
  \frac
  {
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{x_\theta}(x_t, t)
  
  }
  {1 - \bar{\alpha}_{t}}
  
  
  -
  
  
  \frac
  {
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
  
  }
  {1 - \bar{\alpha}_{t}}
  ||^2_2]
  
  \\=
  
  argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
  \frac
  {
  \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) 
  
  }
  {1 - \bar{\alpha}_{t}}
  (
  \hat{x_\theta}(x_t, t)
  -
  x_0
  )
  ||^2_2]
  
  \\=
  
  argmin_\theta \frac{1}{2\sigma_q^2(t)} 
  \frac
  { \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
  {(1 - \bar{\alpha}_{t})^2}
  [||
  
  (
  \hat{x_\theta}(x_t, t)
  -
  x_0
  )
  ||^2_2]
  $$

- 最终，VDM 的优化目标可以表示为：以任意时刻的噪声输入 $x_t$ ，预测原始图像 $\hat{x_0}$ ，并计算其与真实输入的 $x_0$ 的误差。训练过程中， $t$ 从 $[2, T]$ 均匀采样。

## 3.1 Learning Diffusion Noise Parameters

$\alpha_t$ 之前的介绍中，一直是作为提前预设的常量来使用，在DDPM中也是从 $[0.0001, 0.02]$ 线性增加的。然而，$\alpha_t$ 也可以作为一个可学习的参数。如，使用参数为 $\eta$ 的模型，预测第 $t$ 个时间步的 $\alpha_t$ 可以表示为 $\hat{\alpha}_\eta (t)$ 。 

为了优化参数 $\eta$ ，仍然从 $argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))$ 推导：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
\\=

argmin_\theta \frac{1}{2\frac
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}\frac{1 - \bar{\alpha}_{t}}{
(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} (1 - \alpha_t)}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} - \bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
噪声参数 $\alpha_t$ 需要控制随着 $t$  的增加，图像的噪声比例越来越大。本文使用 $SNR = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ 信噪比来计算噪声比例。因此，论文中在推导时，为了构造 $SNR$ 的形式，对上式补了一些项，上式可以继续推导：
$$
=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} - \bar{\alpha}_{t-1}\bar{\alpha}_{t} + \bar{\alpha}_{t-1}\bar{\alpha}_{t} - \bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

argmin_\theta \frac{1}{2}
\frac
{\bar{\alpha}_{t-1} (1 - \bar{\alpha}_{t}) - \bar{\alpha}_{t}(1 - \bar{\alpha}_{t-1})}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]


\\=

argmin_\theta \frac{1}{2}
(
\frac
{\bar{\alpha}_{t-1} (1 - \bar{\alpha}_{t})}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 
-
\frac
{\bar{\alpha}_{t}(1 - \bar{\alpha}_{t-1})}
{(1 - \bar{\alpha}_{t-1}) (1 - \bar{\alpha}_{t})} 
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]


\\=

argmin_\theta \frac{1}{2}
(
\frac
{\bar{\alpha}_{t-1}}
{(1 - \bar{\alpha}_{t-1})} 
-
\frac
{\bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t})} 
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
刚才描述的 $SNR = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ ，实际上是由 $SNR$ 的定义 $SNR = \frac{\mu^2}{\sigma^2}$ ，以及 $q(x_t|x_0) \sim N(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t))$ 得到的：
$$
SNR = \frac{\mu^2}{\sigma^2}
\\=

\frac{\bar{\alpha}_tx_0^2}{1 - \bar{\alpha}_t}
$$
其中，$x_0^2$ 由于在输入图像确定时，后续加噪的过程中都不变，因此可以忽略。

最终，
$$
\frac{1}{2}
(
\frac
{\bar{\alpha}_{t-1}}
{(1 - \bar{\alpha}_{t-1})} 
-
\frac
{\bar{\alpha}_{t}}
{(1 - \bar{\alpha}_{t})} 
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]

\\=

\frac{1}{2}
(
SNR(t-1)
-
SNR(t)
)

[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
根据SNR的含义：

- 更高的SNR表示更多原始信号
- 更低的SNR表示更多的噪声
- 这里不是以dB为单位，即没有加log，因此SNR一定为正。

因此，随着 $t$ 的增加， $SNR$ 需要单调递减。定义 $w_\eta(t)$ 的输出是单调递增的，则 $-w_\eta(t)$ 的输出是单调递减的。为了保证 SNR 非负，使用 exp 进行映射：
$$
SNR(t) = exp(-w_\eta (t))
$$
即，
$$
\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t} = exp(-w_\eta (t))
$$
可以得到：
$$
\alpha_t = \frac{exp(-w_\eta(t))}{1+exp(-w_\eta(t))}
$$
由于 exp 一定为正数，因此 $0 \lt \frac{exp(-w_\eta(t))}{1+exp(-w_\eta(t))} \lt 1$ 。因此，$\alpha_t$ 可以使用 Sigmoid 来限制范围，即：
$$
\bar{\alpha_t} = sigmoid(-w_\eta(t))
\\
1 - \bar{\alpha}_t  = sigmoid(w_\eta(t))
$$

 ## 3.2 三种等价形式

### 3.2.1 形式一：预测原始图像

该形式即为上述推导的形式。

### 3.2.1 形式二：预测噪声

在之前的推导中，最终得到了优化目标为：
$$
argmin_\theta \frac{1}{2\sigma_q^2(t)} 
\frac
{ \bar{\alpha}_{t-1} (1 - \alpha_t)^2 }
{(1 - \bar{\alpha}_{t})^2}
[||

(
\hat{x_\theta}(x_t, t)
-
x_0
)
||^2_2]
$$
即，计算输入图像，和重构的输入图像的误差。除此之外，还有其他几种等价形式。

根据：
$$
x_t =
\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
可以得到：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
$$
把 $x_0$ 带入到 $\mu_q(x_t, x_0)$ ：
$$
\mu_q(x_t, x_0) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}

}
{1 - \bar{\alpha}_{t}}


\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
(1 - \alpha_t) \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\alpha}_t}

}
{1 - \bar{\alpha}_{t}}


\\=

\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}} 
x_t

+

\frac
{1 - \alpha_t}
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t} 
x_t
-

\frac
{ (1 - \alpha_t) \sqrt{1 - \bar{\alpha}_t} }
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t}
\bar{z}_{0}

\\=

(
\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}} 


+

\frac
{1 - \alpha_t}
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t} 
)
x_t
-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}


\\=

(
\frac
{\alpha_t (1 - \bar{\alpha}_{t-1})}
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t})} 


+

\frac
{1 - \alpha_t}
{(1 - \bar{\alpha}_{t})\sqrt{\alpha}_t} 
)
x_t
-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}

\\=


\frac
{
\alpha_t (1 - \bar{\alpha}_{t-1})
+
1 - \alpha_t
}
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t})} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}


\\=


\frac
{ 1 -\bar{\alpha}_{t} }
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t})} 
x_t
-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}

\\=


\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}
$$
同理，我们可以把模型预测的 $\mu_\theta (x_t, t)$ 表示为：
$$
\mu_\theta (x_t, t) =

\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{\theta}(x_t, t)
$$
这种情况下，模型优化的目标为：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [
||

(
\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{\theta}(x_t, t)
)

-

(\frac
{ 1 }
{\sqrt{\alpha_t}} 
x_t

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0})
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [
||


\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{0}

-

\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
\bar{z}_{\theta}(x_t, t)

||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [
||


\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}
(\bar{z}_{0} - \bar{z}_{\theta}(x_t, t))

||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)}
\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t} [
||



(\bar{z}_{0} - \bar{z}_{\theta}(x_t, t))

||^2_2]
$$
其中，$\bar{z}_{\theta}(x_t, t)$ 是神经网络预测的噪声 $\bar{z}_0 \sim N(0, 1)$ ，表示从 $x_0$ 一步到 $x_t$ 的噪声。上述过程证明了，模型预测原始的图像 $x_0$ 与预测噪声是完全等价的。

此外，一些工作发现，预测噪声的效果会更好。

### 3.2.2 形式三：Score-based

由于
$$
x_t = 

\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}

\sim N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)
$$
根据 Tweedie's Formula ： 
$$
E[\mu_{x_t} | x_t] = x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t)) = \sqrt{\bar{\alpha}_t} x_{0}
$$
为了简单起见，后续把 $\nabla_{x_t} log(p(x_t))$ 简写成 $\nabla log(p(x_t))$

因此，
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
之前推导过：
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
把 $x_0$ 带入得到：
$$
\mu_q(x_t, x_0) = 
\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{

\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
+ 
(1 - \alpha_t) \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\alpha}_t}

}
{1 - \bar{\alpha}_{t}}

\\=

\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t}
{1 - \bar{\alpha}_{t}}

+

\frac
{(1 - \alpha_t) x_t}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}

+

\frac
{(1 - \alpha_t) (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}

\\=

(
\frac
{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
)
x_t


+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{\alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{\alpha_t  - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{1 - \bar{\alpha}_{t}}
{\sqrt{\alpha}_t (1 - \bar{\alpha}_{t})}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

\\=
\frac
{1}
{\sqrt{\alpha}_t}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))
$$
但是，实际数据分布 $p(x_t)$ 是不可能知道的，因此使用神经网络来进行预测 $s_\theta(x_t, t)  \approx  \nabla_{x_t} log(p(x_t))$ 。即：
$$
\mu_q(x_t, x_0) =

\frac
{1}
{\sqrt{\alpha}_t}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
s_\theta(x_t, t)
$$
因此，最终的优化问题为：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{1}
{\sqrt{\alpha}_t}
x_t

+

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
s_\theta(x_t, t)

-

\frac
{1}
{\sqrt{\alpha}_t}
x_t

-

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))

||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} [||
\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
s_\theta(x_t, t)

-

\frac
{(1 - \alpha_t)}
{\sqrt{\alpha}_t}
\nabla_{x_t} log(p(x_t))
||^2_2]

\\=

argmin_\theta \frac{1}{2\sigma_q^2(t)} 
\frac
{(1 - \alpha_t)^2}
{\alpha_t}
[||

s_\theta(x_t, t)

-
\nabla_{x_t} log(p(x_t))
||^2_2]
$$
然而，$\nabla_{x_t} log(p(x_t))$ 是并不知道的。回过头再看 $x_0$ 的表达式：
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
其中，$x_t$ 是初始噪声， $\nabla_{x_t} log(p(x_t))$ 是对数似然的梯度方向，对应函数增长最快的方向，沿着梯度方向，就可以使似然函数增大。按照这个观点， $(1 - \bar{\alpha}_t)$ 就可以类比为步长/学习率。

即：从随机初始化的高斯噪声 $x_t$ 开始，沿着 $x_t$ 的对数似然的梯度方向进行更新，使 $x_t$ 的对数似然不断增大，就可以获得真实的数据分布 $p(x)$ 。这个过程也等价于从随机噪声 $x_t$ 开始，每步逐渐去噪。该流程也可以用数学定义严格证明：

由于:
$$
x_t = 

\sqrt{\bar{\alpha}_t} x_{0} 
+
\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
所以：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
$$
对比基于对数似然的梯度的表达式：
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
可以得出：
$$
(1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t)) = - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
$$
即：
$$
\nabla_{x_t} log(p(x_t)) = -  \frac{\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{(1 - \bar{\alpha}_t) }
$$
即：

$x_t$ 沿着对数似然的梯度方向更新，对应于按照一定的比例减去 $\frac{\sqrt{1 - \bar{\alpha}_t} }{(1 - \bar{\alpha}_t) }$ 噪声。

-----

下面按照论文中的介绍，来解释 Score-based 观点：

![image-20230530112721720](imgs/2-Understanding%20Diffusion%20Models%20A%20Unified%20Perspective/image-20230530112721720.png)

对于变量 $x_i$ ，我们想让其对数似然增大，可以让 $x_i$ 向着其对数似然的梯度方向进行更新：
$$
x_{i+1} \gets x_i + c \nabla log(p(x_t)) + \sqrt{2c} z \quad , i =0,1,...K
$$
其中， $x_0$ 使从先验概率分布中随机采样得到的， $z \sim N(z; 0, 1)$ 是随机噪声项。

然而，如上图所示，在概率分布的低密度区域，计算对数似然的梯度是非常不准确的（因为低密度区域存在样本的概率接近为0，而0的log是未定义的）。



# 4 Guidance

目前为止我们只关注数据的分布 $p(x)$ ，然而，我们通常更关注 $p(x|y)$ 来根据条件信息 $y$ 控制图像的生成。

一个自然的方法是向添加时间编码一样，把条件 $y$ 的编码也在每个timestep添加进去，即：
$$
p(x_{0:T}|y) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t,y)
$$
比如，$y$ 可以是文本编码，或低分辨率的图像。因此，此时模型预测值和真值的关系为：$\hat{x}_\theta(x_t, t, y) \approx x_0$ ，$\hat{\epsilon}_\theta(x_t, t, y) \approx \epsilon_0$ ，或 $s_\theta(x_t, t, y) \approx \nabla log(p(x_t|y))$ 。

然而，使用这种方法训练的模型，可能不足以学习到给定的条件信息，甚至会忽略条件信息。因此需要更加显示的提供控制信息，两种主流方法分别为 Classifier Guidance 和 Classifier-Free Guidance 。

## 4.1 Classifier Guidance

为了便于理解，这里使用 score-based 方法来介绍。加入条件控制之后，$\nabla log(p(x_t))$ 就变成了 $\nabla log(p(x_t|y))$  ：
$$
\nabla log(p(x_t|y)) = \nabla log(\frac{p(x_t)p(y|x_t)}{p(y)})
\\=
\nabla log(p(x_t)) + \nabla log(p(y|x_t)) - \nabla log(p(y))
$$
由于计算的梯度与 $x_t$ 有关，因此 $\nabla log(p(y)) = 0$ 

其中，

- $\nabla log(p(x_t))$ 是原始的Diffusion的形式
- $\nabla log(p(y|x_t))$ 表示增加了分类器的梯度

为了更细粒度的控制生成部分和条件控制部分，可以对两部分梯度进行加权：
$$
\nabla log(p(x_t|y)) = \nabla log(p(x_t)) + \gamma \nabla log(p(y|x_t))
$$

- 当 $\gamma = 0$ ，表示完全忽略条件控制

Classifier Guidance 需要单独训练一个分类器，由于该分类器需要处理各种噪声输入，因此无法直接使用预训练好的分类器，所以必须要与Diffusion模型一起训练。

## 4.2 Classifier-Free Guidance

Classifier Guidance 中，存在 $ \nabla log(p(y|x_t))$ 需要单独训练一个分类器，为了去除这一项：
$$
\nabla log(p(y|x_t)) = \nabla log(\frac{p(y)p(x_t|y)}{p(x_t)})
\\=
\nabla log(p(y)) + \nabla log(p(x_t|y)) - \nabla log(p(x_t))
\\=
\nabla log(p(x_t|y)) - \nabla log(p(x_t))
$$
带入 $\nabla log(p(x_t|y))$ 得到：
$$
\nabla log(p(x_t|y)) = \nabla log(p(x_t)) + \gamma (\nabla log(p(x_t|y)) - \nabla log(p(x_t)))
\\=
\gamma \nabla log(p(x_t|y)) + (1 - \gamma) \nabla log(p(x_t))
$$
$\gamma$ 同样是控制条件信息和对数似然的因子：

- 当 $\gamma = 0$ ，完全忽略控制条件
- 当 $\gamma = 1$ ，完全通过引导生成模型
- 当 $\gamma \gt 1$ ，Diffusion模型不仅增强conditional score的学习，同时还把梯度方向向着 unconditional score的反方向移动。这样降低了样本的多样性。
