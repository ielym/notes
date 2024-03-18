# 1 生成模型

对于观测数据 $x$ ，生成模型的目标是学习建模一个真实数据分布 $p(x)$ ，从而能够根据 $p(x)$ 生成新的样本。

## 1.1 问题定义

学习真实数据分布 $p(x)$ 的目标就是获取 $p(x)$ 分布的参数 $\theta$ ，因此生成模型的任务可以定义成参数求解问题。通常有两种方法进行参数估计：

- **频率派视角：**频率派认为 $\theta$ 是一个确定值，因此需要用一些参数估计方法把 $\theta$ 求解出来。通常用极大似然估计的方法，定义似然函数为 $p(x|\theta)$ ，最优化目标为 $\theta^* = argmax logp(x|\theta)$ ，利用已经观测到的训练数据来极大化对数似然，使用最优化如剃度下降法进行求解。
- **贝叶斯视角：**贝叶斯派认为 $\theta$ 不是一个确定值，而是认为 $\theta$ 本身也是服从某种分布的一个随机变量。求解过程通常是从已经观测到的样本中求解后验概率分布$p(\theta|x)$ ，并利用最大后验法 $\theta ^* = argmax logp(\theta|x)$ 进行求解。

VAE，HVAE，扩散模型都可以从贝叶斯视角进行解释和问题定义，因此只关注贝叶斯派中的后验概率分布 $p(\theta|x)$ 求解。

> 为了更好的理解后验概率分布，可以先从上帝视角看看VAE，扩散模型等大概都是怎么做的。根据贝叶斯公式，后验概率分布 $p(\theta|x)$ 可以表示成：
> $$
> p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}
> $$
>
> - VAE中：
>
>   - 似然函数 $p(x|\theta)$ 作为 Decoder，用极大似然估计的方式求解（梯度下降法）。
>   - 先验概率分布 $p(\theta)$ 人为定义成标准高斯分布
>   - $p(x)$ 难以求解
>
> - 扩散模型中：
>
>   - $p(x|\theta)$ 被定义成了一个马尔科夫过程，即 $p(x|\theta) = p(x_0|x_T) = p(x_T) \prod_{t=1}^T p(x_{t-1}|x_t)$ ，对于 $p(x_{t-1}|x_t)$ 这里先不做过多讨论，后面扩散模型中会根据马尔科夫链的性质给出其分布函数，如：
>     $$
>     q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
>     \\ \sim \frac{
>     N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_{0} , 1 - \bar{\alpha}_{t-1}) 
>     N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
>     }
>     {N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)}
>                                 
>     \\ \propto
>     exp\{
>     -[
>                                 
>     \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{2 (1 - \alpha_t)}
>     +
>     \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{2 (1 - \bar{\alpha}_{t-1})}
>     +
>     \frac{(x_t - \sqrt{\bar{\alpha}_t} x_{0})^2}{2(1 - \bar{\alpha}_t)}
>     ]
>     \}
>     $$
>     
>
>   - 先验概率分布 $p(x_T)$ 也被人为定义成了标准高斯分布
>
>   - $p(x)$ 难以求解

从上面两个例子中可以看出，求解后验概率分布都需要真实数据分布 $p(x)$ ，无法人为定义，也难以对真实数据分布作出合理的假设，下一节会介绍针对 $p(x)$ 难以求解的两种不同的处理手段。

## 1.2 $p(x)$

直接求解真实数据分布 $p(x)$ 是比较困难的，难以求解的原因如下：

> $p(x)$ 是真实概率分布，未知。所以先看看能不能利用概率论中的一些定义绕开直接求解 $p(x)$ ：
>
> - 根据边缘概率分布：
>   $$
>   p(x) = \int_z p(x, z) dz
>   $$
>
> - 根据贝叶斯公式：
>   $$
>   p(x) = \frac{p(z) p(x|z)}{p(z|x)}
>   $$
>
> - 根据能量模型：
>   $$
>   p(x) = \frac{1}{Z_\theta} f(x, \theta)
>   $$
>   
>
> 难以求解的原因为：
>
> - 边缘概率分布中，如果 $z$ 的维度较大，则需要 $dz_1 dz_2 dz_3, ....dz_n$ ，难以求解。
> - 贝叶斯公式中，分母有一个后验概率 $p(z|x)$ ，是我们最终想要求解的目标，死循环。
> - 能量模型中，归一化常量 $z_\theta = \int_x  q(x; \theta) dx$ ，当 $x$ 的维度较大，则需要 $dx_1dx_2dx_3, ...dx_n$ ，也是难以求解的。

至此，由于 $p(x)$ 难以求解想要直接求解最初的目标（后验概率分布）好像是一个不可能完成的任务了。但可以用 ELBO 的方式来直接近似求解后验概率分布 $p(z|x)$ 

## 1.3 ELBO

需要注意，ELBO 不是用来求解/估计 $p(x)$ 的，而是用来直接估计后验概率分布 $p(z|x)$ 的。既然是直接估计真实后验分布，那就需要用另外一个分布去近似 $p(z|x)$ ，用来近似真实后验分布的概率分布定义成 $q_\phi (z|x)$ 。为了衡量 $q_\phi (z|x)$ 和 $p(z|x)$ 的相似程度，可以用距离函数 $d(\cdot, \cdot)$ 来表示，ELBO 中使用 KL散度作为距离函数 $d$ 。即，最终的目标是最小化KL散度：
$$
KL(q_{\phi}(z|x) || p(z|x)  = \int_z q_{\phi}(z|x) log (\frac{q_{\phi}(z|x)}{p(z|x)}) dz  = \sum_z q_{\phi}(z|x) log (\frac{q_{\phi}(z|x)}{p(z|x)}) = E[log (\frac{q_{\phi}(z|x)}{p(z|x)}]
$$
然而，既然要计算KL散度，但真实后验概率分布 $p(z|x)$ 还是未知的，ELBO的解决方法如下：

> - 首先，用根据贝叶斯公式，后验概率分布可以表示为：
>   $$
>   p(z|x) = \frac{p(x, z)}{p(x)}
>   $$
>
> - 替换 KL 散度中的后验概率分布，并化简：
>   $$
>   KL(q_\phi (z|x)|| p(z|x))
>                 
>   \\= \int_z  q_\phi (z|x) log(\frac{p(x)q_{\phi}(z|x)}{p(x,z)}) dz
>                 
>   \\= \int_z  [q_\phi (z|x) [log(p(x)) + log(\frac{q_\phi (z|x)}{p(x,z)})] dz 
>                 
>   \\=  \int_z  q_\phi (z|x)log(p(x))dz + \int_z  q_\phi (z|x) log(\frac{q_\phi (z|x)}{p(x,z)})dz
>                 
>   \\= log(p(x)) \int_z [q_{\phi}(z|x)dz + E[log(\frac{q_\phi (z|x)}{p(x, z)})]
>   $$
>
> - 其中，$q_{\phi}(z|x)$ 是概率密度函数，积分是1，因此上式最终可以写为：
>   $$
>   KL(q_\phi (z|x)|| p(z|x)) = log(p(x)) + E[log(\frac{q_\phi (z|x)}{p(x, z)})]
>   $$
>
> - 对于对数观测似然（证据） $log(p(x))$ ，虽然不知道其分布到底是什么，但是只要当训练数据给定时，$p(x)$ 就是一个未知但确定的常量了，所以 $log(p(x))$  可以看作是一个固定值。把 $log(p(x))$ 移动到一边：
>   $$
>   log(p(x)) = KL(q_\phi (z|x)|| p(z|x)) - E[log(\frac{q_\phi (z|x)}{p(x, z)})]
>   \\= KL(q_\phi (z|x)|| p(z|x)) + E[log(\frac{p(x, z) }{q_\phi (z|x)})]
>   \\
>   \ge E[log(\frac{p(x, z) }{q_\phi (z|x)})] = ELBO
>   $$

由于 $log(p(x))$ 是固定值常量，所以最小化两个后验概率分布 $q_\phi (z|x)$ 和 $p(z|x)$  的KL散度，等价于最大化 ELBO。ELBO中的两个概率分布都比较好求/好学习，后面在VAE，扩散模型中分别介绍处理方法（分子都是根据 $p(x, z) = p(z) p(x|z)$ 进行处理；分母在VAE中是Decoder，在扩散模型中是人为定义好的加噪过程的马尔科夫链）。

## 1.4 VAE

VAE的默认公式中，直接最大化ELBO，这种方法是变分的。对于 ELBO 中的 $p(x, z)$ ，VAE 利用链式法则进行处理：
$$
p(x, z) = p(z) p(x|z)
$$
因此ELBO变成：
$$
E[log(\frac{p(x, z) }{q_\phi (z|x)})] 
\\=
E[log(\frac{p(z) p(x|z)}{q_\phi (z|x)})]
\\=
E[log(p(x|z))] + E[log(\frac{p(z)}{q_\phi (z|x)})]
\\=
E[log(p(x|z))] + KL(p(z) || q_\phi (z|x))
\\=
E[log(p(x|z))] - KL(q_\phi (z|x) || p(z))
$$
从上式可以看出：

- 首先，学习到一个中间的bottleneck，输出分布是 $q_\phi (z|x)$ ，作为 encoder，把输入变换成一个隐空间的概率分布。
- 同时，学习一个确定性的函数 $p(x|z)$ 作为 decoder，把隐向量 $z$ 转变成 $x$ 。 

理解：

- 第一项是重建项，实际使用mse loss
- 第二项是先验分布匹配项，encoder预测均值和方差，并使其与标准正态分布对齐，实际使用的是多元高斯分布的kl损失，推导略。

## 1.5 HVAE

HVAE是VAE的推广，具有多个层级的隐变量。每个隐变量都可以把所有之前的其他隐变量当作条件，但我们只关注一种特殊的 Markovian HVAE (MHVAE)。在MHVAE中，生成过程是一个马尔科夫链，因此解码 $z_t$ 时，只依赖于前一个隐编码 $z_{t+1}$ 。直观上，这个过程可以看作时堆叠多个VAE。

对于 ELBO 中的：

- $p(x, z) = p(x, z_{1:T}) = p(z_T) p_\theta(x|z_1) \prod_{t=2}^T p_\theta (z_{t-1} | z_t)$
- $q_\phi (z|x) = q_\phi (z_{1:T}|x) = q_\phi(z_1 | x) \prod_{t=2}^{T} q_\phi(z_t | z_{t-1})$

此时ELBO可以表示成：
$$
E[log(\frac{p(x, z) }{q_\phi (z|x)})] 
\\=
E[log(\frac{p(x, z_{1:T})}{q_\phi (z_{1:T}|x)})]

\\=

E[log( \frac{p(z_T) p_\theta(x|z_1) \prod_{t=2}^T p_\theta (z_{t-1} | z_t)} {q_\phi(z_1 | x) \prod_{t=2}^{T} q_\phi(z_t | z_{t-1})} )]
$$

# 2 扩散模型 - 变分推断角度

> 变分推断（Variational Inference, VI）是贝叶斯近似推断方法中的一大类方法，将后验推断问题巧妙地转化为优化问题进行求解。即，利用 ELBO 求解后验概率分布。

## 2.1 从HMVAE 转换到变分扩散模型

从HVAE到Variational Diffusion Model，需要添加三个关键约束条件：

- 隐变量的维度与数据维度完全相同
- 每个时间步的encoder （$z_{t-1} \to z_{t}$）不是通过学习得到的，而是预先定义的线性高斯模型。即，每一步生成隐变量的过程是确定的，且都是在上一步的基础上添加高斯噪声。需要注意，每一步 $x_t$ 的均值和方差是不固定的，均值是以上一步的输出为中心（$x_t \in N(\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I$）
- 最终 $T$ 时间步的隐编码是标准高斯分布。

> - 从第一个约束条件，MHVAE对应的encoder可以改写成：
>   $$
>   q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t | x_{t-1})
>   $$
>
> - 从第二个约束条件，每个时间步的编码器不是学习得到的，而是固定的线性高斯模型，其均值和方差可以是预先设置的超参，或预先学习好的超参。这里人为定义超参，第 $t$ 步的均值 $\mu_t (x_t) = \sqrt{\alpha_t} x_{t-1}$ ，方差 $\Sigma_t (x_t) = (1 - \alpha_t)I$ 。
>
>   - 可以看出，均值是一定会随着 $x_{t-1}$ 的变化而变化的
>
>   - 而如果不同时间步的 $\alpha_t$ 相同，则不同时间步的 $x_t$ 的方差也是相同的，因此，选择上述高斯分布的均值方差的形式的目的是确保不同时间步的隐变量的方差基本在一个大致相同的尺度，即 variance-preserving，保方差。
>
>   - 为了提升灵活性，$\alpha_t$ 也可以随着时间步而变化。
>
>   - 每个时间步的encoder $q(x_t | x_{t-1})$ 可以定义为：
>     $$
>     q(x_t | x_{t-1}) = N(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)I)
>     $$
>     即，
>     $$
>     x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon
>     $$
>     其中，$\epsilon \in N(\epsilon; 0, I)$ 
>
> - 从第三个约束条件，$\alpha_t$ 无论是固定的还是一种策略变化的，最终 $x_T$ 的分布都是标准高斯分布，即：
>   $$
>   p(x_T) = N(x_T; 0, I)
>   $$
>
>   - 这是由于 $x_t = \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1}} x_{0}  \sqrt{1 - \alpha_t\alpha_{t-1} \alpha_{t-2}...\alpha_{1}} \bar{z}_{0} =
>     \sqrt{\bar{\alpha}_t} x_{0}  + \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}$ ，当 $\alpha_t < 1$ 且 $T$ 较大时，$x_0$ 的系数接近0，高斯噪声的系数接近1
>
> 可以看出，当基于HMVAE做了上述三个假设之后：
>
> - 编码器的分布 $q(x_t | x_{t-1})$ 的参数不再是 $\phi$ ，而是在每个时间步定义的均值和方差。
> - 因此，VDM中，只需要研究 $p_\theta (x_{t-1} | x_t)$ ，因此可以从高斯噪声生成新数据。

## 2.2 基于ELBO引出变分扩散模型的目标函数

变分扩散模型中，ELBO为：
$$
E[log(\frac{p(x, z) }{q_\phi (z|x)})] 

=

E[log(\frac{p(x, z_{1:T}) }{q_\phi (z_{1:T}|x)})]

=

E[log(\frac{p(x_{0:T}) }{q (x_{1:T}|x_0)})]

=

E[log( \frac{p(x_T) \prod_{t=1}^T p_\theta (x_{t-1}| x_t) }{\prod_{t=1}^T q(x_t|x_{t-1})})]

\\=

E[log(p_\theta (x_0|x_1))] 
-
E[KL(q(x_T|x_{T-1})||p(x_T))] 
-
\sum_{t=1}^{T-1} E[KL( q(x_t|x_{t-1})  || p_\theta (x_t | x_{t+1}))]
$$

> 在VDM中，ELBO被分解成了三项：
>
> - $E[log(p_\theta (x_0|x_1))] $ ：重建项。给定第一步的隐变量 $x_1$ ，预测原始数据 $x_0$ 。
> - $E[KL(q(x_T|x_{T-1})||p(x_T))] $ ：先验匹配项。让最后一步的隐变量的分布匹配正态分布的先验。该项不含任何需要优化的参数，当 $T$ 足够大时，该项自然是0
> - $E[KL( q(x_t|x_{t-1})  || p_\theta (x_t | x_{t+1}))]$ ：一致性项。确保加噪和去噪的分布一致性，即从噪声图像去噪时的分布需要与从干净图像加噪时的分布对应。可以通过训练 $p_\theta (x_t | x_{t+1})$ 去匹配 $q(x_t | x_{t-1})$ 来最小化该项。
>
> 优化VDM的开销主要在第三项上，因为需要优化所有的时间步 $t$ 。**此外，第三项需要用 $\{ x_{t-1}, x_{t+1} \}$ 两个随机变量来计算期望，其蒙特卡洛估计可能会比只用一个随即变量计算具有更大的方差，这是由于需要对 $T-1$ 个一致性项求和，当 $T$ 较大时，估计的最终的 ELBO 的方差可能也会更大。**
>
> 为了让每一项的期望计算都只使用一个随机变量，需要用到关键的公式（马尔可夫链中，$x_t$ 只依赖于 $x_{t-1}$ ，而与 $x_0$ 无关：
> $$
> q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0)
> 
> =
> 
> \frac{q(x_{t-1}|x_t, x_0) q(x_t|x_0)}{q(x_{t-1}|x_0)}
> $$

用 $q(x_t | x_{t-1}, x_0)$ 代替 $q(x_t | x_{t-1})$ ，重新带入 ELBO ，可以得到：
$$
E[log(\frac{p(x_{0:T}) }{q (x_{1:T}|x_0)})] =

E[(log(p_\theta(x_0|x_1)))]
-
KL(q(x_T|x_0)||p(x_T))
-
\sum_{t=2}^{T} E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]
$$

> 至此，成功的推导了ELBO的一种解，并且可以用较低的方差进行估计，因为每一项都最多只计算一个随机变量的期望。这个公式该有一个优雅的解释：
>
> - $E[(log(p_\theta(x_0|x_1)))]$ ：重构项，该项可以**使用蒙特卡洛估计进行近似和优化**。
> - $KL(q(x_T|x_0)||p(x_T))$ ：表示最终的噪声与标准高斯先验的接近程度。该项没有可学习参数，并且基于假设，该项总有 $x_T \in N(0, I)$ ，即该项等于0
> - $\sum_{t=2}^{T} E[KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))]$ ：去噪匹配项。我们学习去噪过程 $p_\theta (x_{t-1} | x_t)$ 作为真实去噪过程 $q(x_{t-1}|x_t, x_0)$ 的近似值。

然而，上述根据 ELBO 推导出来的目标函数还存在问题：

- 对于任意时刻的 $x_t$ ，都需要从 $x_0$ 逐步加噪获得，复杂度增加
- $p_\theta(x_{t-1}|x_t))$ 是网络学习得到的，但是 $q(x_{t-1}|x_t, x_0)$ 却是未知的

> 对于第一个问题，由于不同时刻的隐变量都是线性高斯模型，因此可以得到：
> $$
> x_t = 
> \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}...\alpha_{1}} x_{0} 
> + 
> \sqrt{1 - \alpha_t\alpha_{t-1} \alpha_{t-2}...\alpha_{1}} \bar{\epsilon}_{0}
> 
> \\
> =
> \sqrt{\bar{\alpha}_t} x_{0} 
> +
> \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}_{0}
> 
> \sim N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)
> $$

> 对于第二个问题，展开 $q(x_{t-1}|x_t, x_0)$  ：
> $$
> q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t) = \frac{q(x_{t-1}|x_0)q(x_t|x_{t-1},x_0)}{q(x_t|x_0)}
> \\ \sim \frac{
> N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_{0} , 1 - \bar{\alpha}_{t-1}) 
> N(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
> }
> {N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)}
> 
> \\ \propto
> exp\{
> -[
> 
> \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{2 (1 - \alpha_t)}
> +
> \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_{0})^2}{2 (1 - \bar{\alpha}_{t-1})}
> +
> \frac{(x_t - \sqrt{\bar{\alpha}_t} x_{0})^2}{2(1 - \bar{\alpha}_t)}
> ]
> \}
> 
> \\=
> exp
> \{
> -\frac{1}{2}
> 
> (
> \frac
> {1 - \bar{\alpha}_{t}}
> {(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
> )
> [
> x_{t-1}^2
> -2
> \frac
> {
> 
> \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})x_t
> + 
> \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) x_0
> 
> }
> {1 - \bar{\alpha}_{t}}
> 
> x_{t-1}
> ]
> \}
> $$

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
对照对应可知：
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

> 可以发现:
>
> - $q(x_{t-1}|x_t, x_0)$ 的均值与 $t, x_t, x_0$ 有关
> - 方差只与 $\alpha_t$ 有关，而 $\alpha_t$ 要么在每个时间步都是固定的，要么是作为一组超参数，要么是提前学习好的网络对当前时间步的推理输出，总之与时间 $t$ 有关。
> - 因此，为了后续表示的简单性，把均值方差分别表示为关于 $x_t, x_0$ 或 $t$ 的函数，即  $\mu_q(x_t, x_0),  \sigma_q(t)$

真实值 $q(x_{t-1}|x_t, x_0) = q(x_{t-1}|x_t)$ 已知了之后，就可以让网络预测的 $p_\theta (x_{t-1}| x_t)$ 去进行匹配：

- 把 $p_\theta (x_{t-1}, x_t)$ 也建模成高斯分布，均值为 $\mu_\theta, \sigma_\theta$
- 由于gt $q(x_{t-1}|x_t, x_0)$ 的方差只与 $t$ 有关，不随 $x_0, x_t$ 变化，因此在对齐预测值和真实值时，不用考虑对齐方差。 
- 由于 $p_\theta (x_{t-1}| x_t)$  的条件只有 $x_t$ 而没有 $x_0$ ，因此其均值只是 $x_t$ 的函数，表示成 $\mu_\theta (x_t, t)$ 

根据两个多元高斯分布的KL散度计算公式，最终可以得到目标函数：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
$$
由于网络拟合的是 $p_\theta (x_{t-1}| x_t)$ ，是用来生成 $x_{t-1}$ 的，并不是用来预测均值的。为了获得预测均值，可以对照 gt 的均值的形式获得预测的均值：
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
所以，最终的目标函数为：
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
最终，VDM 的优化目标可以表示为：以任意时刻的噪声输入 $x_t$ ，预测原始图像 $\hat{x_0}$ ，并计算其与真实输入的 $x_0$ 的误差。

## 2.3 预测 $x_0$ 和预测噪声

### 2.3.1 预测 $x_0$ 

在 `2.2` 小节中已经推导过了预测 $x_0$ 的方式：
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
但是预测 $x_0$ 存在不稳定的问题：

- 如果从 $x_T$ 根据马尔科夫过程得到 $x_0$ ，资源占用过大
- 如果用 $x_t = \sqrt{\bar{\alpha}} x_0 + \sqrt{1 - \bar{\alpha}} \epsilon$ ，则 $x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}} \hat{\epsilon}}{\sqrt{\bar{\alpha}}}$ ，由于 $\bar{\alpha} = \alpha_0 \alpha_1 ... \alpha_t$ ，通常数值较小，导致 $\epsilon$ 预测值出现任何扰动，都会的使得估计的 $x_0$ 的波动范围较大，数值不稳定。
- BTW，为了解决从 $\epsilon$ 计算假 $x_0$ 数值不稳定的问题，谷歌在渐进式蒸馏的工作中尝试让网络同时输出 $\epsilon$ 和 $x_0$ ，即输出通道数 $\times 2$ ，发现效果也还可以，比直接预测 $x_0$ 要好一些。

### 2.3.2 预测 $\epsilon$

根据：
$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
$$
带入到  $\mu_q(x_t, x_0)$  可以得到：
$$
\mu_q(x_t, x_0) = 

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
同理：
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


因此，用 $x_0$ 计算目标函数的方式可以替换成：
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
=
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

argmin_\theta \frac{1}{2\sigma_q^2(t)}
\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t} [
||



(\bar{z}_{0} - \bar{z}_{\theta}(x_t, t))

||^2_2]
$$
一些工作发现，预测噪声的效果会更好。

### 3.2.3 预测 v

![image-20231214161710676](./imgs/2-Understanding Diffusion Models A Unified Perspective_v3/image-20231214161710676.png)

从向量的角度理解 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ ，等价于 $x_t$ 是由向量 $x_0, \epsilon$ 线性组合得到的：

- 后面为了简单起见，用 $\alpha_t$ 表示 $\sqrt{\bar{\alpha}_t}$ ，用 $\sigma_t$  表示  $\sqrt{1 - \bar{\alpha}_t}$ 。即：
  $$
  x_t = \alpha_t x_0 + \sigma_t \epsilon
  $$

- 根据扩散模型加噪的人为设置，线性组合的权重具有 $\sqrt{\bar{\alpha}_t ^ 2 + (1 - \bar{\alpha}_t) ^ 2} = \sqrt{\alpha^2 + \sigma^2} =  1$ 的特点。

从上图可以看出，$x_t$ 和 $x_0$ 向量的夹角 $\phi$ 可以表示为：
$$
\phi = arctan \frac{\sigma_t}{\alpha_t}
\\
tan \phi = \frac{\sigma_t}{\alpha_t}
$$

---

先不要看上图，$\alpha$ 和 $\beta$ 一定可以组合成一个向量：$\beta = \alpha + \sigma$ ，且：

- $\alpha = \beta cos\phi$ 

- $\sigma = \beta sin\phi$  

- 不管上图，由于一定有 $\sqrt{\alpha^2 + \sigma^2} =  1$ ，所以也一定有

  - $\alpha = cos\phi$
  - $\sigma = sin\phi$ 

- 所以，不管上图，由于  $x_t = \alpha_t x_0 + \sigma_t \epsilon$ ，如果把 $\alpha_t, \sigma_t$ 都换成 $\phi$的函数，则$x_t$ 也是 $\phi$的函数，即：
  $$
  x_\phi = cos(\phi)x_0 + sin(\phi)\epsilon
  $$
  

---

$v$ 是人为定义的，定义为：
$$
v_{\phi} = \frac{d x_{\phi}}{d \phi} = -sin(\phi)x_0 + cos(\phi) \epsilon
$$

- $v$ 类似于 $dx / dt$ ，即在 $\phi$ 作为自变量的情况下，$x_\phi$ 在 $\epsilon$ 和 $x_0$ 之间每一步的位移，即 $d x_\phi = [-sin(\phi)x_0 + cos(\phi) \epsilon]d\phi$ 

---

如果预测 $v$ ，则采样的时候有（DDPM）
$$
sin(\phi) x_0 = cos(\phi) \epsilon - v_{\phi} 

\\=

cos(\phi) \frac{x_\phi - cos(\phi)x_0}{sin(\phi)} - v_{\phi}

\\=

\frac{cos(\phi)}{sin(\phi)} (x_\phi - cos(\phi)x_0) - v_{\phi}
$$
$\phi$ 乘到左边去：
$$
sin^2 (\phi) x_0 = cos(\phi) x_t - cos^2(\phi) x_0 - sin(\phi)v_{\phi}

\\ => \\

(sin^2 (\phi) + cos^2(\phi)) x_0 = cos(\phi) x_\phi - sin(\phi)v_{\phi}

\\ => \\

x_0 = cos(\phi) x_\phi - sin(\phi)v_{\phi}
$$
同理，想要获得噪声时，也可以得到：
$$
\epsilon = sin(\phi) x_{\phi} + cos(\phi) v_{\phi}
$$

## 2.4 损失权重

### 2.4.1 计算 $\mu$ 的损失权重

从最原始的目标函数开始，计算 $q(x_{t-1}|x_t, x_0)$ 和 $p_\theta(x_{t-1}|x_t)$ 的KL散度，等价于匹配 $\mu$ （DDPM对任务进行了简化，方差 $\sigma^2 =
\frac
{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}
{1 - \bar{\alpha}_{t}}$ 只和 $t$ 有关，本身就是已经对齐了）
$$
argmin_\theta KL(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))
\\=
argmin_\theta \frac{1}{2\sigma_q^2(t)} [||\mu_\theta - \mu_q||^2_2]
$$

- $\sigma(t)^2$ 随时间的变化为：

  ![image-20240107010246753](imgs/2-Understanding%20Diffusion%20Models%20A%20Unified%20Perspective_v3/image-20240107010246753.png)

- 此时目标函数的权重为 $\frac{1}{2\sigma_q^2(t)}$ 

- 由于当 $t$ 越大时，$\sigma$ 越大，则 $\frac{1}{2\sigma_q^2(t)}$ 越小。即，该损失权重更关注于早期时间步的优化。

### 2.4.2 计算 $\epsilon$ 的损失权重

之前推导过，计算 $\epsilon$ 时的目标函数为：
$$
argmin_\theta \frac{1}{2\sigma_q^2(t)}
\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t} [
||



(\bar{\epsilon}_{0} - \bar{\epsilon}_{\theta}(x_t, t))

||^2_2]
$$
首先分析上式（原始）的损失权重 $\frac{1}{2\sigma_q^2(t)}
\frac
{ 1 - \alpha_t  }
{\sqrt{1 - \bar{\alpha}_t}\sqrt{\alpha}_t}$ 随时间 $t$ 的变化为：

![image-20240107010611842](imgs/2-Understanding%20Diffusion%20Models%20A%20Unified%20Perspective_v3/image-20240107010611842.png)

- 当 $t$ 越大，权重越小
- 从头训练时，使用权重可能导致后期时间步无法得到充分学习，因此DDPM的损失函数去掉了权重。也相当于做了一个reweight，保证不同时间步的损失权重都是 $1$ 。

即，DDPM中实际使用的损失函数的权重为 $1$ ：
$$
argmin_\theta  [
||



(\bar{\epsilon}_{0} - \bar{\epsilon}_{\theta}(x_t, t))

||^2_2]
$$

### 2.4.3 计算 $x_0$ 的损失权重

从DDPM的噪声损失中，把 $\epsilon$  替换成：
$$
\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}
$$
可以推导出与权重为 $1$ 的噪声预测损失等价的 $x_0$ 的损失：
$$
argmin_\theta  [
||

\frac{1}{\sqrt{1 - \bar{\alpha}_t}}(x_t - \sqrt{\bar{\alpha}_t}x_0)
- 
\frac{1}{\sqrt{1 - \bar{\alpha}_t}}(x_t - \sqrt{\bar{\alpha}_t}x_0(x_t, t))
||^2_2]

\\=

argmin_\theta  [
||

\frac{\sqrt{\bar{\alpha}_t}}{\sqrt{1 - \bar{\alpha}_t}}
(x_0 - x_0(x_t, t))
||^2_2]

\\=

argmin_\theta  [
\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
||


(x_0 - x_0(x_t, t))
||^2_2]
$$

- 可以看出，计算噪声等价于计算 $x_0$ 加上一个权重。

定义：
$$
SNR = \frac{\mu^2}{\sigma^2} = \frac{\bar{\alpha}_t x_0}{1 - \bar{\alpha}_t}
\\ \approx
\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

- 约等于把 $x_0$ 去掉的 SNR，是因为对于同一个 $x_0$ 的不同时间步的信噪比，$x_0$ 是常量。

因此，预测 $x_0$ 的目标函数可以表示为：
$$
argmin_\theta  [
SNR(t)
||


(x_0 - x_0(x_t, t))
||^2_2]
$$

### 2.4.4 计算 $v$ 的损失权重






# 3 从变分推断角度到分数匹配角度

根据 Tweedie's Formula，从 $x_t = \sqrt{\bar{\alpha}_t} x_{0} \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0} \sim N(x_t; \sqrt{\bar{\alpha}_t} x_{0} , 1 - \bar{\alpha}_t)$ 可以得到：
$$
E[\mu_{x_t} | x_t] = x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t)) = \sqrt{\bar{\alpha}_t} x_{0}
$$
即：
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$

> 上式可以理解为：
>
> - $x_t$ 是初始噪声
> - $\nabla_{x_t} log(p(x_t))$ 是对数似然的梯度方向，对应函数增长最快的方向，沿着梯度方向，就可以使似然函数增大。
> -  $(1 - \bar{\alpha}_t)$ 就可以类比为步长/学习率。
>
> 即：从随机初始化的高斯噪声 $x_t$ 开始，沿着 $x_t$ 的对数似然的梯度方向进行更新，使 $x_t$ 的对数似然不断增大，就可以获得真实的数据分布 $p(x)$ 。

> 这个过程也等价于从随机噪声 $x_t$ 开始，每步逐渐去噪，证明如下：
>
> 由于：
> $$
> x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{\sqrt{\bar{\alpha}_t}}
> $$
> 对照：
> $$
> x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
> $$
> 可以得出：
> $$
> (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t)) = - \sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}
> $$
> 即：
> $$
> \nabla_{x_t} log(p(x_t)) = -  \frac{\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{(1 - \bar{\alpha}_t) }
> $$
> 即：
>
> $x_t$ 沿着对数似然的梯度方向更新，对应于减去一定的比例 $\frac{\sqrt{1 - \bar{\alpha}_t} }{(1 - \bar{\alpha}_t) }$ 的噪声。

在刚才的目标函数推导中，本来预测的是 $\mu$ ，但是由于不知道 gt 的均值，所以替换成了 $x_0$ 或 $\epsilon$ 。现在有了分数 $\nabla_{x_t} log(p(x_t))$ ，也可以用分数来替换均值，从而计算损失函数为：

把
$$
x_0 = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} log(p(x_t))}{\sqrt{\bar{\alpha}_t}}
$$
带入到
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
可以化简出：
$$
\mu_q(x_t, x_0) 
=
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
因此可以使用神经网络来进行预测 $s_\theta(x_t, t)  \approx  \nabla_{x_t} log(p(x_t))$ 。即：
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
=
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
其中，gt 是刚才推导的：
$$
\nabla_{x_t} log(p(x_t)) = -  \frac{\sqrt{1 - \bar{\alpha}_t} \bar{z}_{0}}{(1 - \bar{\alpha}_t) }
$$

# 4 扩散模型 - 分数匹配角度

> 基于分数模型的整体时间线为：
>
> - 为了避免求解能量模型 $p(x;\theta) = \frac{1}{Z(\theta)} q(x; \theta)$ 中的归一化常量 $z_\theta$ ，[Estimation of non-normalized statistical models by score matching] 论文提出了 score matching 方法。通过计算分数 $\nabla_x p_\theta(x)$ ，巧妙的把 $z_\theta$ 通过求导的方式给消掉了。
> - 在生成模型中，即使把归一化常量 $z_\theta$ 通过score matching的方式消掉了，但是还是无法获得 gt 的 $p_\theta(x)$ ，也就无法训练。从而引出了 Denoising score matching 的方法，把未知的数据分布 $p_\theta(x)$ 转移到预先定义的已知分布 (如高斯分布) $q_\sigma(\tilde{x} | x)$ 上去，之后用 $s(\tilde{x};\theta)$ 去学习 $q_\sigma(\tilde{x} | x)$ ，即 $J(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}} ||^2]$ 。
> - 就算学习到了对数数据分布的梯度，如何采样？通常用朗之万动力学进行采样 $\tilde{x}_t =  \tilde{x}_{t-1} + \frac{\epsilon}{2} \nabla_x logp(\tilde{x}_{t-1}) + \sqrt{\epsilon} z_t$ 。
> - 然而，在 Denoising score matching 为代表的 score-based 的生成模型中存在几个问题（流形假说，低数据密度区域），因此Song yang 提出了 Noise Conditional Score Networks （NCSN），在解决了现有问题之外，通过多个水平的高斯噪声对数据进行扰动，获得一系列收敛与真实分布的噪声扰动的数据分布，利用这些中间分布可以提高多峰采样的准确率。由于 NCSN 用朗之万动力学进行采样，因此该方法也称为 Score matching with Langevin dynamics (SMLD) 。
> - 最后，Song yang 用 SDE 的角度，把 DDPM, SMLD等方法统一到了同一个框架下。

## 4.1 Score Matching

> 对于真实数据分布 $p_{data}(x)$ ，我们想要用具有参数 $\theta$ 的概率密度函数 $p(x, \theta)$ 来对其近似。概率密度函数用能量模型表示为：
> $$
> p(x;\theta) = \frac{1}{Z(\theta)} f(x; \theta)
> $$
> 其中，分子称为能量函数；分母称为归一化常量。
>
> 根据概率密度函数积分为1的特性，可以得到归一化常量的表达式 $Z(\theta) = \int_x  f(x; \theta) dx$  ，当 $x$ 的维度较大时，就几乎不可能计算出积分了。

Score Matching 就是一种避免计算 $z_\theta$ 的一种方法。具体的：

- 发现如果对概率密度函数计算对数梯度，则可以巧妙的把归一化常量消掉：
  $$
  \nabla_x log(p_\theta(x)) = \nabla_x f_\theta(x)
  $$

- 因此，现在的目标就是建模 score function $s(x;\theta)$ ，令其与观测似然 $log(p_\theta(x))$ 一致即可，从而引出最小二乘法的目标函数：
  $$
  \theta^* = arg min_\theta \frac{1}{2} \mathbb{E}_{x} [|| s(x;\theta) -  \nabla_x log(p_\theta(x))||^2]
  $$

- 然而，为了计算目标函数，还是不知道数据分布 $p_\theta(x)$ 。score matching 在基于一些弱规则条件下，把目标函数进一步表示成了：
  $$
  J(\theta) = \int_x p(x) \sum_{i=1}^n [\partial_i s_i(x;\theta) + \frac{1}{2} s_i(x;\theta)^2] dx + C
  =
  \mathbb{E}_x [\frac{1}{2}|| s_i(x;\theta) ||^2 + \nabla_x^2 s_i(x;\theta)  ||^2]
  $$
  此时只需要只需要对数能量函数的一阶导和二阶导即可求解参数 $\theta$ 。

## 4.2 Denoising Score Matching

用Score Matching的方法可以在不考虑归一化常量的情况下实现参数估计，对于已知的概率密度函数（如高斯分布），只需要考虑能量函数的对数的梯度即可（高斯函数右边exp的内容）。但是，对于生成任务，能量函数都是未知的，自然也就无法利用 score matching 进行参数估计。为解决该问题，出现了 Denoising score matching 的方法：

- 由于图像的数据分布 $p_\theta(x)$ 未知，也就没法计算 $\nabla_x log(p_\theta(x))$ 来监督分数预测网络。但是我们可以给图像加上一些噪声，强迫图像变成我们预定义的数据分布（如高斯分布 $q_\sigma(\tilde{x} | x)$） 。

- 之后可以在加噪之后的已知数据分布上学习分数预测网络：
  $$
  J(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}} ||^2]
  $$

- 但是，这种方法的局限性比较明显，只有当添加的噪声较小时，才会有 $s_\theta^*(x) = \nabla_x log q_\sigma(x) \approx \nabla_x log p_{data}(x)$ ，才可以采样出 $q_\sigma(x) \approx p_{data}(x)$ 

## 4.3 Sampling with Langevin dynamics

之前一直在说可以预测概率密度函数的对数的梯度，即分数。但预测分数之后怎么生成/采样新的数据？可以利用朗之万动力学进行采样。朗之万动力学可以仅使用score function $\nabla_x log(p(x))$ 就能实现在概率密度函数 $p(x)$ 中采样，采样公式为：
$$
\tilde{x}_t =  \tilde{x}_{t-1} + \frac{\epsilon}{2} \nabla_x logp(\tilde{x}_{t-1}) + \sqrt{\epsilon} z_t
$$

> - $\epsilon > 0$ 是步长
> - $z_t \in N(0, I)$ 是标准布朗运动。
> - 初始值 $\tilde{x}_0 \in \pi(x)$ ，$\pi$ 是先验分布，如自己添加的预设好的噪声数据分布。
> - 当 $\epsilon \to 0, T \to \infty$ 时，在一些正则条件下，$\tilde{x}_T$ 会收敛到服从 $p_{data}(x)$ 的分布。反过来说，当 $\epsilon > 0$ 并且 $T < \infty$ 时，Metropolis-Hastings（MH）更新需要修正拉格朗日动力学的误差，但是这个误差通常可以忽略。本文假设当$\epsilon$ 很小并且 $T$ 较大时，该误差可以忽略不计。

## 4.4 SMLD / NCSN

> NCSN和SMLD是同一个工作，NCSN是论文中提出的分数预测网络的名称 (Noise Conditional Score Networks, NCSN)，而 SMLD 是采样的过程 (Score matching with Langevin dynamics, SMLD) ，即 NCSN + 退火朗之万动力学采样
>
> SMLD主要是为了解决Score-based生成模型存在的一些问题：
>
> - **流形假说**。不管是真实图像数据，还是神经网络提取的特征，都是比较稀疏的，即数据分布仅存在于整个编码空间中的一部分（低维流形），并没有占满整个编码空间。所以就有两个问题：
>
>   - 由于 $p_{data}(x)$ 在低维流形中可能没有分布，所以分数 $\nabla_x logp_{data}(x)$ 在低维流形中也是未定义的。
>   - 之所以能够根据目标函数 $J(\theta) = \frac{1}{2} \mathbb{E}_{x} [|| s(x;\theta) -  \nabla_x log(p_\theta(x))||^2]$ 来进行最优化求解，得到 $\theta^* = \theta$ ，是基于样本占满了整个编码空间的前提推出来的，当数据仅分布在低维流形时，score matching的目标函数结论就可能不适用了。
>
> - **低数据密度区域**。就算样本占满了整个空间，但是数据分布可能不均匀，存在一些低密度的数据分布区域，给 score matching 的估计和MCMC采样带来一定困难：
>
>   - 由于低密度区域缺失样本，因此学习到的分数模型 $s_\theta(x)$ 在低密度区域上无法准确的估计$logp_{data}(x)$ 的梯度。
>
>   - 当真实数据分布是混合数据分布时，如两个混合高斯模型，可以推导出两个高斯分布的分数中把混合比例 $\pi$ 消掉了，所以用朗之万采样时，采样出来的样本的分布也会和 $\pi$ 无关，不符合真实数据分布。如下图所示：
>
>     - 图 (a) 是真实的数据分布，采样到左下角的数据的概率应该低于采样到右上角的数据的概率
>     - 图 (b) 直接用真实的数据分布来计算score，并用朗之万动力学进行采样。可以看出，即使用真实的score，采样出的数据分布也和真实数据分布不一致，两簇数据的概率几乎相同，并不符合真实数据的情况。
>
>     ![image-20231218154602915](./imgs/2-Understanding Diffusion Models A Unified Perspective_v3/image-20231218154602915.png)

SMLD中，作者认为用随机高斯噪声扰动数据能够使数据更适合基于分数的生成模型：

- 由于高斯噪声是分布在整个编码空间的，因此扰动数据不存在流形假设中所说的真实数据仅存在于低维流形中的问题。
- 高斯噪声还能够填充原始未扰动数据分布中的低密度区域，从而能够进行更准确的分数估计。
- 此外，利用多个水平的高斯噪声进行扰动，能够获得一系列收敛于真实数据分布的噪声扰动的数据分布，利用这些中间分布，可以提高多峰采样（即混合概率分布）的准确率。

基于上述直觉，本文对基于分数的生成模型进行了以下改进：

- 使用不同level的噪声扰乱数据
- 训练一个单个的，有条件（噪声水平）的score估计网络，来估计所有不同噪声水平下的分数。
- 采样时，使用训练时最大的噪声的分数作为初始值，并逐渐退火降低噪声水平。这种方式有助于把大噪声水平的优势平稳的转移到低噪声水平。

---

NCSN的训练流程：

- 对于原始数据 $x$ ，使用 $L$ 个方差逐渐减小的 $\sigma_1 > \sigma_2 > ... > \sigma_L$ 高斯噪声对数据进行扰动，获得一系列具有不同噪声水平的加噪数据分布 $q_\sigma(\tilde{x} | x) \in N(\tilde{x}; x, \sigma_i^2I)$ ，其中单个噪声水平下的分数模型的目标函数为：
  $$
  J(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}} ||^2]
  $$

  - 由于噪声分布 $q_\sigma(\tilde{x} | x)$ 都是人为预先定义好的高斯噪声，所以 $\frac{\partial log(q_\sigma(\tilde{x}| x))}{\partial \tilde{x}}$ 也是已知的，并且可以化简成：

    ![image-20231218161834934](./imgs/2-Understanding Diffusion Models A Unified Perspective_v3/image-20231218161834934.png)

  - 最终的目标函数是 
    $$
    l(\theta; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\tilde{x} - x}{\sigma^2}  ||^2]
    $$

- 由于不同噪声水平的图像分布不一样，因此需要 $L$ 个分数预测网络，成本较大。所以 NCSN 把噪声水平当作额外的控制条件，通过 IN 把条件 $\sigma_i$ 注入到分数预测网络中。由于一个网络需要学习多个不同噪声水平下的图像分布，因此需要对不同噪声水平的损失进行加权：
  $$
  L(\theta; \{ \sigma_i \}_{i=1}^L) = \frac{1}{L} \sum_{i=1}^{L} \lambda(\sigma_i) l(\theta;\sigma_i)
  $$
  

  - 实验发现，目标函数中的 $||s(...)||_2 \propto 1/\sigma$ 


  - 因此，启发式的设置 $\lambda(\sigma_i) = \sigma^2$ ，从而实现：
    $$
    l(\theta; \sigma) = \frac{1}{2} \sigma^2 \mathbb{E}_{\tilde{x}} [|| s(\tilde{x};\theta) -  \frac{\tilde{x} - x}{\sigma^2}  ||^2]
    \\=
    l(\theta; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{x}} [|| \sigma s(\tilde{x};\theta) -  \frac{\tilde{x} - x}{\sigma}  ||^2]
    $$


  - 这样就同时实现了：

    - $||\sigma s(...)||_2 \propto 1$
    - $\frac{\tilde{x} - x}{\sigma} \in N(0, I)$ 


  - 这样做的好处是，对于不同水平的 $\sigma$ ，loss是完全在同一个水平的，因为 loss中的两项与 $\sigma$ 的大小没有任何关系。

---

SMLD的推理流程：

![image-20231218163654239](./imgs/2-Understanding Diffusion Models A Unified Perspective_v3/image-20231218163654239.png)

退火Langevin dynamics算法如上图所示：

- 从固定的先验分布开始，初始化 $\tilde{x}_0$ 。
- 从分布 $q_{\sigma_1}(x)$ 开始，获得步长 $\alpha_1$ 
- 对于每个噪声水平，运行 $T$ 个Langevin dynamics采样步，采样出 $q_{\sigma_2}(x)$ 中的分布。
- 之后降低步长到 $\alpha_2$   

由于每一步的数据分布 $\{ q_{\sigma_i} \}_{i=1}^L$ 都是受到一定程度的高斯干扰的数据，因此这些分布能够填满整个空间，并且每个分布的分数都是很好的定义的。

# 5 扩散模型 - SDE角度

> 目前为止有两种比较成功的扩散模型，SMLD 和 DDPM。这两种扩散模型都是在原始分布未知的情况下，通过给原始数据添加噪声，转换到一种人为预先定义的已知分布上进行求解的方法：
>
> - **Score matching with Langevin dynamics (SMLD)** 。在每个噪声尺度估计数据概率密度函数的梯度，之后用朗之万动力学来降低噪声等级。
> - **Denoising diffusion probabilistic modeling (DDPM)** 。训练一系列概率模型来反转每一步的噪声。
>
> 如第3小节和第4小节所介绍，SMLD和DDPM都可以看作是基于分数的生成模型。但是 SMLD 和 DDPM 的加噪/去噪过程都有明显的差别，使用 SDE 的视角，可以把之前的扩散模型都统一到同一个框架下，便于进一步的分析和优化。

## 5.1 从朗之万动力学推导出 SDE 的标准形式

> 推导之前，站在上帝视角，SDE标准形式：$dx = f(x, t) dt + g(t) dw$ 

朗之万动力学：
$$
x_{t+1} = x_t + \epsilon \nabla_x logp(x_t)+ \sqrt{2 \epsilon} z_i
$$
当 $K \to \infty$ 时，定义 $\Delta t = \epsilon$ ，$\Delta t \to 0$ ：
$$
x_{t+1} - x_t = \Delta t \nabla_x logp(x_t)+ \sqrt{2 \Delta t} z_i
$$
把 $\nabla_x logp(x_t)$ 替换成 $f(x, t)$ ，把 $\sqrt{2}$ 替换成 $g(t)$ , 则：
$$
x_{t + \Delta t} - x_t = f(x, t) \Delta t + g(t) \sqrt{\Delta t} z_i
$$
其中，
$$
\sqrt{\Delta t} z_i \in N(0, \Delta t I)
$$
这里引入布朗运动，则：
$$
w_{t + \Delta t} = w_t + N(0, \Delta t I)
\\=
w_t + \sqrt{\Delta t} z_i
$$
即：
$$
w_{t + \Delta t} - w_t = \sqrt{\Delta t} z_i
$$
代入布朗运动并代入，得到：
$$
x_{t + \Delta t} - x_t = f(x, t) \Delta t + g(t) (w_{t + \Delta t} - w_t)
$$
当 $\Delta t \to 0$ 时：
$$
dx = f(x, t) d t + g(t)dw
$$

## 5.2 用 SDE 加噪

目标是用一系列连续的时间变量 $t \in [0, T]$ 来重建一个扩散过程 $\{ x(t) \}_{t=0}^T$ 。其中，$x(0) \in p_0$ 是独立同分布的观测数据中的样本，$x(T) \in p_T$ 是先验分布（如高斯噪声）。扩散过程可以用 Ito 的SDE形式表示如下：
$$
dx = f(x, t) dt + g(t) dw
$$

- $w$ 是一个标准维纳过程（如，布朗运动）
- $f(\cdot, t)$ 称作 $x(t)$ 的漂移系数，实现 $\mathbb{R}^d \to \mathbb{R}^d$ 的映射。
- $g(\cdot)$ 是实值函数，称作 $x(t)$ 的扩散系数，是一个标量：$\mathbb{R} \to \mathbb{R}$ 。为了简单起见，我们假设扩散系数是一个标量，而不是 $d \times d$ 的矩阵，并且不依赖于 $x$ 
- $dw$ 表示一个很小的高斯白噪声。

## 5.3 用 SDE 去噪

从 $x(T) \in p_T$ 开始，并进行逆向 SDE，可以最终获得 $x(0) \in p_0$ 。之前的工作证明扩散过程的逆过程仍然是一个扩散过程，逆向 SDE 的表达式为：
$$
dx = [f(x, t) - g(t)^2 \nabla_x logp_t(x)] dt + g(t) d\bar{w}
$$
其中，$\bar{w}$ 是一个标准维纳过程。$dt$ 是一个负无穷小的时间步。一旦知道每个时间步的边缘分布的对数梯度 $\nabla_x logp_t(x)$ 后，就可以用SDE逆向过程生成 $p_0$ 。

> 推导如下：
>
> 根据 SDE 前向加噪过程，可以得到：
> $$
> x(t + \Delta t) = x(t) + dx = x(t) + f(x, t) dt + g(t) dw
> $$
> 其中，$g(t) dw = g(t) dt z(t) \in N(0, g(t)^2 dtI)$ 。
>
> 因此 SDE 加噪过程的条件概率分布为：
> $$
> p(x_{t+\Delta t} | x_t) = N(x_{t+\Delta t}; x_t + f(x, t) d t, g(t)^2 d t I) 
> \\ 
> \propto
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) d t ||^2}{2g(t)^2 d t})
> $$
> 根据贝叶斯公式，可以得到 SDE 逆向去噪过程的概率分布：
> $$
> p(x(t) | x(t + \Delta t)) = \frac{p(x_{t + \Delta t} | x_t) p(x_t)}{p(x_{t + \Delta t})}
> \\=
> p(x_{t + \Delta t}) exp(logp(x_t) - logp(x_{t+\Delta t}))
> 
> \\ \propto
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} + logp(x_t) - logp(x_{t + \Delta t}))
> $$
> 其中，对 $logp(x_{t + \Delta t})$ 做泰勒展开（$p(x_t)$ 是双变量 $x_t, t$ 的函数，所以展开式有两项）：
> $$
> logp(x_{t + \Delta t}) \approx logp(x_t) + (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) + \Delta t \frac{\partial logp(x_t)}{\partial t}
> $$
> 把泰勒展开代入，得到逆向过程的分布：
> $$
> p(x(t) | x(t + \Delta t))  \propto
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} + logp(x_t) - logp(x_{t + \Delta t}))
> \\=
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} + logp(x_t) - logp(x_t) - (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) - \Delta t \frac{\partial logp(x_t)}{\partial t}
> 
> \\=
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} - (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) - \Delta t \frac{\partial logp(x_t)}{\partial t}
> $$
> 由于 $\Delta t \to 0$ ，所以上式中最后一项 $\to 0$ ，即：
> $$
> p(x(t) | x(t + \Delta t))  \propto
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} - (x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t) - \Delta t \frac{\partial logp(x_t)}{\partial t}
> 
> \\ \approx
> exp(- \frac{|| x(t + \Delta t) - x_t - f(x, t) \Delta t ||^2}{2g(t)^2 \Delta t} - \frac{2g(t)^2 \Delta t(x_{t + \Delta t} - x_t) \nabla_{x_t} logp(x_t)}{2g(t)^2 \Delta t})
> 
> \\=
> exp(- \frac{|| x_{t + \Delta t} - x_t - [f(x, t) - g(t)^2 \nabla_{x_t} logp(x_t) ] \Delta t ||^2}{2g(t)^2 \Delta t})
> $$
> 由于当 $\Delta t \to 0$ 时，$t + \Delta t \to t$ 。同时由于逆向过程中，$t + \Delta t$ 是已知的，而 $t$ 未知，因此用 $t + \Delta t$ 替换 $t$ ：
> $$
> p(x(t) | x(t + \Delta t)) \approx
> exp(- \frac{|| x_{t + \Delta t} - x_t - [f(x, t) - g(t)^2 \nabla_{x_t} logp(x_t) ] \Delta t ||^2}{2g(t)^2 \Delta t})
> 
> \\=
> 
> exp(- \frac{|| x_{t} - x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] \Delta t ||^2}{2g(t + \Delta t)^2 \Delta t})
> $$
> 即：
> $$
> p(x(t) | x(t + \Delta t))  \in N(x(t); x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] \Delta t, g(t + \Delta t)^2 \Delta t I)
> $$
> 由于逆向SDE中，$dx = x_{t + \Delta t} - x_t$ ，并且其中的 $x_t$ 可以根据 $p(x(t) | x(t + \Delta t))$ 进行采样：
> $$
> x_t = x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] \Delta t + g(t + \Delta t) \sqrt{\Delta t} w
> \\=
> x_{t + \Delta t} - [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] dt + g(t + \Delta t) dw
> $$
> 因此：
> $$
> dx = x_{t + \Delta t} - x_t 
> \\=
> [f(x_{t + \Delta t}, t + \Delta t) - g(t + \Delta t)^2 \nabla_{x_{t + \Delta t}} logp(x_{t + \Delta t}) ] dt - g(t + \Delta t) dw
> $$
> 再取 $\Delta t \to 0$ ：
> $$
> dx = x_{t + \Delta t} - x_t 
> \\=
> [f(x_{t}, t) - g(t)^2 \nabla_{x_{t}} logp(x_{t}) ] dt - g(t) dw
> $$
>
> 标准的SDE逆向过程是：
> $$
> dx = [f(x_{t}, t) - g(t)^2 \nabla_{x_{t}} logp(x_{t}) ] dt + g(t) dw
> $$
> 正负号应该对于采样高斯噪声没有影响。

## 5.4 把 SMLD/DDPM 统一到 SDE 框架下

### 5.4.1 VE SDE (SMLD)

> **Variance Exploding (VE) ：**方差爆炸，对于 SMLD，扩散公式为 $x_T = x_0 + \sigma_T \epsilon$ ，需要加一个特别大的方差，才能让 $X_T$ 变成高斯噪声，因此称作方差爆炸。

原始的离散形式下，SMLD加噪公式为：
$$
x_t = x_0 + \sigma_t \epsilon \in N(x_0, \sigma_t^2)
$$

$$
x_{t-1} = x_0 + \sigma_{t-1} \epsilon \in N(x_0, \sigma_{t-1}^2)
$$

想要逐步的向连续靠，就要想办法用 $x_{t-1}$ 表示 $x_t$ ：
$$
x_t \in N(x_0, \sigma_t^2 ) = N(x_0, \sigma_t^2 - \sigma_{t-1}^2 +  \sigma_{t-1}^2)
\\=
x_{t-1} + N(0, \sigma_t^2 - \sigma_{t-1}^2) 
\\=
x_{t-1} + \sqrt{\sigma_t^2 - \sigma_{t-1}^2} \epsilon
$$
其中，$t = 1,2,...,N$ 。当 $N \to \infty$ 时，马尔科夫链 $\{ x_t \}_{t=1}^N$ 可以近似看成是连续的 $\{ x(t) \}_{t=0}^1$。如果把上式看作连续的，符号可以稍微改一下：
$$
x_{t + \Delta t} = x_{t} + \sqrt{\sigma_{t+\Delta t}^2 - \sigma_{t}^2} \epsilon
\\=
x_{t} + \sqrt{\frac{\sigma_{t+\Delta t}^2 - \sigma_{t}^2}{\Delta t}} \sqrt{\Delta t} \epsilon
\\=
x_{t} + \sqrt{\frac{\Delta \sigma_t^2}{\Delta t}} \sqrt{\Delta t} \epsilon
$$
当 $\Delta t \to 0$ 时，且用离散的符号替换，有：
$$
x(t+\Delta t) - x(t) = \sqrt{\frac{\Delta \sigma(t)^2}{\Delta t}} \sqrt{\Delta t} \epsilon
\\
dx = \sqrt{\frac{d [\sigma(t)^2]}{d t}} \sqrt{\Delta t} \epsilon
\\=
\sqrt{\frac{d [\sigma(t)^2]}{d t}} d \epsilon
$$
其中，$t \in [0, 1]$ ，而不再是 $1, 2, ..., i, ..., N$ ，因此 $t = i / N$ 。

对照SDE标准形式 $dx = f(x, t) dt + g(t) dw$ ，可以确定：

- $f(x_t, t) = 0$
- $g(t) = \sqrt{\frac{d [\sigma(t)^2]}{d t}}$
- $w = \epsilon \in N(0, I)$ 

### 5.4.2 VP SDE (DDPM)

> Variance Preserving (VP) ：方差缩紧，对于DDPM，扩散公式为 $x_T = \sqrt{\bar{\alpha}_T} x_0 + \sqrt{1 - \bar{\alpha}_T} \epsilon$ ，主要依靠较小的 $\sqrt{\bar{\alpha}_T}$ 来压制原始图像，并且每一步添加的噪声方差为 $\sqrt{1 - \bar{\alpha}_T}$ 不算大，因此称作方差缩进。

和SMLD类似，还是先考虑离散情况：
$$
x_i = \sqrt{1 - \beta_i} x_{i-1} + \sqrt{\beta_i} z_{i-1}
$$
其中，$i = 1, 2, ..., N$ 。

个人理解，为了凑成SDE的标准形式，所以论文里定一个辅助变量，令 $\{ \bar{\beta}_i = N \beta_i \}_{i=1}^N$ 。因此上式可以表示成：
$$
x_i = \sqrt{1 - \frac{\bar{\beta}_i}{N}} x_{i-1} + \sqrt{\frac{\bar{\beta}_i}{N}} z_{i-1}, i=1,2,...,N
$$
并且，当 $N \to \infty$ 时，定义连续的 $\beta(t)$ 是辅助变量，即：
$$
\beta(t) = \{ \bar{\beta}_i \}_{i=1}^N , t \in [0, 1]
$$
 因此，

-  $t = \frac{i}{N}$ 

-  $\beta(\frac{i}{N}) = \bar{\beta}_i$ 
-  $x(\frac{i}{N}) = x_i$
-  $z(\frac{i}{N}) = z_i$ 
-  $\Delta t = \frac{1}{N}$ ，$t \in \{ 0, 1, ..., \frac{N - 1}{N} \}$ 

所以上式可以进一步写成：
$$
x(t + \Delta t) = \sqrt{1 - \frac{\bar{\beta}_i}{N}} x(t) + \sqrt{\frac{\bar{\beta}_i}{N}} z(t)
\\=
\sqrt{1 - \frac{\beta(t + \Delta t)}{T}} x(t) + \sqrt{\frac{\beta(t + \Delta t)}{T}} z(t)
\\=
\sqrt{1 - \beta(t + \Delta t) \Delta t} x(t) + \sqrt{\beta(t + \Delta t) \Delta t} z(t)
$$
由于当 $x \to 0$ 时，$(1 - x)^{\alpha} \approx 1 - \alpha x$ ，所以当 $\Delta t \to 0$ 时，上式可以近似写成：
$$
x(t + \Delta t) = \sqrt{1 - \beta(t + \Delta t) \Delta t} x(t) + \sqrt{\beta(t + \Delta t) \Delta t} z(t)
\\ \approx 
(1 - \frac{1}{2} \beta(t ) \Delta t) x(t) + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$
 对照标准形式 $dx = f(x, t) dt + g(t)dw$ ：
$$
x(t + \Delta t) 
\approx 
(1 - \frac{1}{2} \beta(t ) \Delta t) x(t) + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$

$$
x(t + \Delta t) - x(t)
\approx 
- \frac{1}{2} \beta(t ) \Delta t x(t) + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$

$$
dx
\approx 
- \frac{1}{2} \beta(t ) x(t) dt  + \sqrt{\beta(t) } \sqrt{\Delta t} z(t)
$$

因此：

- $f(x, t) = - \frac{1}{2} \beta(t ) x(t)$
- $g(t) = \sqrt{\beta(t) }$

## 5.5 DDPM的去噪过程等价于 SDE 求解器



## 5.6 统一 DDPM 和 SMLD 的求解器 （PC Sampling）



## 5.7 从 SDE 到 PF-ODE 形式



# 5 一致性模型

## 5.1 CM

## 5.2 LCM

## 5.3 LCM-LoRA

