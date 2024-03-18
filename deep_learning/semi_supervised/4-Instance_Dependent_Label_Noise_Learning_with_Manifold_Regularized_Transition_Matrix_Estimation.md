`Instance-Dependent Label-Noise Learning with Manifold-Regularized Transition Matrix Estimation`

# 1 定义

+ $D$ 表示由样本和标签组成的一组随机变量 $(X, Y)$。
+ $X \in \mathbb{R}^d$ ，$d$ 表示特征维度。
+ $Y = {1, 2, ..., K}$ ，$K$ 表示类别总数。
+ 对于给定的示例 $x \in X$ ，分类问题需要预测出一个标签 $y \in Y$ 。
+ 定义 $\bar{D} := \{(x_i, \bar{y_i})\}_{i=1}^{N}$ 为噪声样本 $(X, \bar{Y})$ 。其中，$\bar{Y}$ 表示噪声标签。
+ 定义 $T(x)$ 为 Instance-dependent transition matrix (IDTM) 。



# 2 方法

由于实际情况中，通常难以获取大量且干净的数据集 $D$ 。$T(x)$ 可以作为干净数据集 $D$ 和 噪声数据 $\bar{D}$ 之前的映射：
$$
P(\bar{Y}=j | X =x) = \sum_{i=1}^{K} T_{ij}(x) P(Y = i | X = x)
$$
其中，$P(\bar{Y}=j | X =x) $ 是噪声样本的类别后验概率，$P(Y = i | X = x)$ 是干净样本的类别后验概率。$T(x) = (T_{ij}(x))_{i,j=1}^{K} \in [0, 1] ^ {K \times K}$ 。

转移矩阵 $T(x)$ 形如 (假设类别总数 $K = 5$ )：
$$
T(x) = 
\begin{bmatrix}
T_{00}(x) & T_{01}(x) & T_{02}(x)  & T_{03}(x)  & T_{04}(x) \\
T_{10}(x) & T_{11}(x) & T_{12}(x)  & T_{13}(x)  & T_{14}(x) \\
T_{20}(x) & T_{21}(x) & T_{22}(x)  & T_{23}(x)  & T_{24}(x) \\
T_{30}(x) & T_{31}(x) & T_{32}(x)  & T_{33}(x)  & T_{34}(x) \\
T_{40}(x) & T_{41}(x) & T_{42}(x)  & T_{43}(x)  & T_{44}(x) \\
\end{bmatrix}
$$
其中，$T_{ij}(x)$ 表示，当 $Y = i$ 时，（下一状态） $\bar{Y} = j$ 的概率。因此，由全概率公式能够得到：
$$
P(\bar{Y} = j | X = x) = P(Y = i | X = x) P(\bar{Y} = j | Y=i, X=x)
$$
对比 $P(\bar{Y}=j | X =x) = \sum_{i=1}^{K} T_{ij}(x) P(Y = i | X = x)$ ，可以发现，转移矩阵 $T_{ij}(x)$ 就等于 $P(\bar{Y} = j | Y=i, X=x)$ 。

因此，本文的目标就是通过准确的估计 IDTM  $T(x)$ ，获得一个分类器，能够从噪声训练数据中训练，并准确的分类测试实例。

然而，在上述公式中，无法获得 $P(Y = i | X = x)$ ，因为无法判断噪声样本所对应的干净标签是什么。唯一可以获得的就只有噪声样本的类别后验概率 $P(\bar{Y} = j | X = x)$ 。需要做的工作包含两步：

+ 提取干净（置信度高, extracted confident clean examples）的样本。
+ 基于干净样本和噪声样本，优化 IDTM $T(x)$ 。



## 2.1 Extract Confident Clean Examples

提取出干净样本是至关重要的，本文采用的是蒸馏法，利用蒸馏法，提取出一个子数据集。

获得干净的子数据集之后，就可以在干净的数据 $D$ 和 噪声数据 $\bar{D}$ 之间来学习转移矩阵 $T(x)$ 。

但是，显然 $D$ 是从含有噪声的数据集 $\bar{D}$ 中蒸馏出来的一个子集，因此还需要从 $\bar{D}$ 中采样出与 $D$ 中样本相同的一个含有噪声的自集 $\bar{D}^s := \{(x_i^s, \bar{y_i})\}_{i=1}^{N^s}$ ，但是为了简单起见，后续默认仍然使用 $D$ 代替 $\bar{D^s}$ 。

此外，虽然 $D$ 是蒸馏出来的干净样本，但是也无法保证其标签都是完全正确的，因此使用 $\hat{y}$ 表示干净样本的标签，而不再是之前使用的 $y$ 了。

至此，重新梳理一下变量表示：

+ 蒸馏出的干净数据集 $D$ ，其中样本的标签用 $\hat{y}$ 表示。
+ 与干净数据集 $D$ 中样本对应的噪声标签数据集 $\bar{D}$ ，其中样本的噪声标签用 $\bar{y}$ 表示。
+ 为了计算 $T(x)$ ，公式为 $P(\bar{Y} = j | X = x) = P(Y = i | X = x) P(\bar{Y} = j | Y=i, X=x) = P(Y = i | X = x) T_{ij}(x)$  ，后续会将 $Y$ 对应替换为 $\hat{Y}$ 。

## 2.2 Framework

需要获取的：

+ 干净数据的后验概率分布 $P(\hat{Y} = j | X = x) $ 。
+ 把 $T(x)$ 作为未知量，通过 $T(x)$ 把干净数据的后验概率分布变成噪声数据的后验概率分布 $P(\bar{Y} = j | x)$ 。

流程如下：

+ 使用干净的数据训练出一个分类器 $f(x; w)$ 。

+ 输入干净数据 $x_i$，得到概率分布 $P(\hat{Y} | x_i;w)$ 。

+ 使用一个网络 (参数为 $\theta$) 预测 $T(x_i; \theta)$ 。

+ 使用 $T(x_i; \theta)$ 和 $P(\hat{Y} | x_i;w)$ ，获得估计的噪声标签 $T(x_i; \theta) P(\hat{Y} | x_i;w)$ 。

+ 实际的噪声标签为 $\bar{y_i}$ 。

+ 使用交叉熵，计算估计的噪声标签和实际噪声标签的损失：
  $$
  - \frac{1}{N} \bar{y_i} log( T(x_i; \theta) f(x_i; w) )
  $$

+ 其中，网络的参数有两部分：预测分类概率分布的 $w$ ，以及预测 $T(x)$ 的 $\theta$ 。

+ 因此，最终的优化目标是使得这两个网络的输出计算出的交叉熵损失最小化，即：
  $$
  \underset{w, \theta}{\min } R(w, \theta) = - \frac{1}{N} \bar{y_i} log( T(x_i; \theta) f(x_i; w) )
  $$

  + 其中， $N$ 是提取出的干净数据集的样本总数。

虽然直观上看，确实可以通过直接优化上述交叉熵损失，来优化 $w$ 和 $\theta$ 。然而，$T(x; \theta)$ 在没有任何假设的情况下非常难以学习，其原因是 $T(x; \theta)$ 的自由度太高，但是 估计噪声 = T*干净分布 这样一个线性系统中，等式的个数和变量的个数相同。

现有方法为了解决该问题，通常利用减少 $T(x)$ 的自由度来降低复杂度，但是这种方式可能会造成估计错误的问题。

相反的，在本文中，没有使用任何的 strong restrictions， 只使用了轻微的假设：

+ 两个实例越相似，其转移矩阵也越相似。即：
  $$
  M_I = \sum_{i,j=1}^{N} S_{ij}^{I} ||T(x_i) - T(x_j)||^2
  $$
  其中，$S_{ij}^{I}$ 是一个指示函数，即 $T(x_i)$ 和 $T(x_j)$ 足够相近时， $S_{ij}^{I} = 1$ ，否则为0。那么如何评价两个转移矩阵足够相近呢？对于样本 $T(x_i)$ ，其 k-nearest neighbour 才是足够相近的 (实验中此处的 $k = 7$)，即 $x_j \in N(x_i, k_1)$ ，$N(x_i, k)$ 表示 k=7-最近邻。但是这样说两个实例相近仍然不充分，因为相邻的实例也可能受各种影响而被标记成不同的类别，两个真正相似的实例应该大概率被错误的标记成同一类，否则就不能算是真正的相似。因此，$S_{ij}^{I}$ 最终的表达形式为：
  $$
  \begin{equation}
  \label{}
  S_{ij}^{I} =\left\{
  \begin{aligned}
  1 & , & if &&  x_j \in N(x_i, k) && and && \bar{y_i} = \bar{y_j}, \\
  0 & , & && else.
  \end{aligned}
  \right.
  \end{equation}
  $$
  这里k近邻使用的欧式距离。

+ 两个实例相似，但是被错误标记成了两个不同的类别，则其转移矩阵也应该具有差异：
  $$
  M_B = \sum_{i,j=1}^{N} S_{ij}^{B} ||T(x_i) - T(x_j)||^2
  $$

  $$
  \begin{equation}
  \label{}
  S_{ij}^{B} =\left\{
  \begin{aligned}
  1 & , & if &&  x_j \in N(x_i, k) && and && \bar{y_i} \ne \bar{y_j}, \\
  0 & , & && else.
  \end{aligned}
  \right.
  \end{equation}
  $$

  实验中，此处的 $k = 7$

+ 显然，最终的目标需要最小化 $M_I$ ，最大化 $M_B$ ，因此该部分的 loss 为：
  $$
  M(\theta) = M_I - M_B
  $$

+ 但是，$S_{ij}^{I}$ 和 $S_{ij}^{B}$ 两个指示函数都采用硬编码的方式，无法有效度量两个样本的相似程度。因此，作者提出了一种更加有效的度量方式：
  $$
  \begin{equation}
  \label{}
  S_{ij}^{I} =\left\{
  \begin{aligned}
  e^{-\frac{||x_i - x_j||^2}{\sigma^2}} & , & if &&  x_j \in N(x_i, k1) && and && \bar{y_i} = \bar{y_j}, \\
  0 & , & && else.
  \end{aligned}
  \right.
  \end{equation}
  $$

  $$
  \begin{equation}
  \label{}
  S_{ij}^{B} =\left\{
  \begin{aligned}
  e^{-\frac{||x_i - x_j||^2}{\sigma^2}} & , & if &&  x_j \in N(x_i, k2) && and && \bar{y_i} \ne \bar{y_j}, \\
  0 & , & && else.
  \end{aligned}
  \right.
  \end{equation}
  $$

  这样可以根据两个样本的欧式距离的真正的相似程度，来确定转移矩阵的权重。同时指数映射也能够放大不同相似程度的样本之间的转移矩阵的权重的差异。$\sigma^2$ 是一个超参。

综上所述，本文最终的loss为：
$$
\underset{w, \theta}{\min } L(w, \theta) = R(w, \theta) + \lambda M(\theta)
$$
其中，超参设置：

+ $\lambda = 0.3$
+ $\sigma = 1.1$

