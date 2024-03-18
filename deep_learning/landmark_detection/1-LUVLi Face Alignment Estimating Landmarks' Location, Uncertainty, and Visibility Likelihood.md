# 2 方法

![image-20230420234452054](imgs/1-LUVLi%20Face%20Alignment%20Estimating%20Landmarks%27%20Location,%20Uncertainty,%20and%20Visibility%20Likelihood/image-20230420234452054.png)

- Backbone : DU-Net
- 添加了三个组件分别用于预测关键点，不确定性 和 可见性：
  - **Mean Estimator** ：用于预测关键点。计算heatmap上所有输出值为整数的加权和来得到坐标位置。
  - **Cholesky Estimator Network (CEN)** ：用于预测不确定性。直接估计多元拉普拉斯分布或高斯分布。
  - **Visibility Estimator Network (VEN)** ：用于预测可见性

- 其中， CEN 和 VEN 的权重在每个 UNet 都是共享的。
- 定义：
  - 第 $i$ 个U-Ne t预测的第 $j$ 个关键点的坐标表示为 $u_{ij}$ ，估计的协方差矩阵 $\Sigma_{ij}$ ，估计的可见性 $\hat{v_{ij}}$，损失 $L_{ij}$
  - 第 $j$ 个关键点的位置坐标的真值为 $p_j$ 。可见性为 $v_j \in \{0, 1\}$ ，$1$ 表示可见，$0$ 表示不可见。

## 2.1 Mean Estimator

关键点位置分支输出尺寸与输入图像尺寸相同，通道数为 $N_p$ （关键点数量）。

定义 $H_{ij}(x, y)$ 表示第 $i$ 个 UNet 上负责预测第 $j$ 个关键点 (第 $j$ 个通道) 的特征图上 $(x, y)$ 位置的特征值。关键点预测值 $u_{ij} = [u_{ijx}, u_{ijy}]^T$  的计算方式为：
$$
u_{ij} = 

\left[ 
\begin{array}{}
	u_{ijx} \\
	u_{ijy}
\end{array}
\right]

 = 
 
 \frac
 {\sum_{x,y} \sigma(H_{ij}(x, y)) \left[ 
\begin{array}{}
	x \\
	y
\end{array}
\right]
} 
{\sum_{x,y} \sigma(H_{ij}(x, y))}
$$
其中，$H_{ij}(x, y)$ 是标量。$\sigma$ 是用于对特征值 $H_{ij}(x, y)$ 做后处理的函数。作者实验了三种不同的 $\sigma$ ：ReLU，Softmax, 带温度系数的Softmax。实验发现ReLU虽然不是最好，但省区了调节温度系数的额外参数：

![image-20230420191348078](imgs/1-LUVLi%20Face%20Alignment%20Estimating%20Landmarks%27%20Location,%20Uncertainty,%20and%20Visibility%20Likelihood/image-20230420191348078.png)

## 2.2 Cholesky Estimation Network (CEN)

不确定性预测实际使用的是MSE来计算预测位置和真值的差异，如 $(p_{jx} - u_{jx})^2$, $(p_{jy} - u_{jy})^2$ 。网络预测不确定性，实际上预测的是不确定性的期望，如 $E[(p_{jx} - \mu_{jx})^2]$ （gt和pred的位置都是一个随机变量），而$E[(p_{jx} - \mu_{jx})^2]$ 实际表示关键点位置的方差。

按照 $x$ 位置的方差来计算 $y$ 的方差，为 $E[(p_{jy} - \mu_{jy})^2]$ 。同时，本文还考虑了 $x$ 和 $y$ 的协方差 $E[(p_{jx} - \mu_{jx})(p_{jy} - \mu_{jy})]$ 。以上三项放在一起，很容易看出就是 $x$ 方向位置的随机变量，与 $y$ 方向位置的随机变量的协方差矩阵：
$$
\Sigma_j =

\left[ 
\begin{array}{}
	E[(p_{jx} - \mu_{jx})^2] & E[(p_{jx} - \mu_{jx})(p_{jy} - \mu_{jy})] \\
	E[(p_{jx} - \mu_{jx})(p_{jy} - \mu_{jy})] & E[(p_{jy} - \mu_{jy})^2]
\end{array}
\right]


\\ = 
\left[ 
\begin{array}{}
	\Sigma_{j_{xx}} & \Sigma_{j_{xy}} \\
	\Sigma_{j_{xy}} & \Sigma_{j_{yy}}
\end{array}
\right]
\\
$$
由于协方差矩阵是对称矩阵，因此 $x, y$ 两个方向的协方差矩阵的自由度只有 $4 - 1 = 3$ （反对角线的两个协方差相同，只需预测出下三角的元素即可）。

![image-20230421011002634](imgs/1-LUVLi%20Face%20Alignment%20Estimating%20Landmarks%27%20Location,%20Uncertainty,%20and%20Visibility%20Likelihood/image-20230421011002634.png)

此外，由协方差矩阵的性质可知，协方差矩阵一定是对称半正定矩阵。而对于实半正定矩阵，可以进行Cholesky分解：
$$
A = LL^T
$$
因此，Cholesky Estimation Network 预测出一个  $N_p \times 3$ 的特征向量 （下三角或上三角） 即可确定 uncertain 协方差矩阵 。 同时，由于协方差矩阵对角线上的元素一定非负，因此作者使用 ELU + 常数来保证该性质。

回过头观察本文提出的Uncertain为什么会以协方差矩阵表示不确定性。可以体会出其思想就是图像的二阶混合中心矩，其中关键点的期望是质心，二阶混合原点矩（协方差）的特征值和最大特征值所对应的特征向量表示预测的Feature-map的分布的主轴方向，即关键点位于主轴方向上的概率较大，如：

![图像矩](imgs/1-LUVLi%20Face%20Alignment%20Estimating%20Landmarks%27%20Location,%20Uncertainty,%20and%20Visibility%20Likelihood/v2-4a73a373810d4baeeaa15e41ce3ca009_720w.jpg)



## 2.3 Location Likelihood

由于本文使用特征图上所有位置的加权和来估计一个关键点的位置，并且关键点存在 **epistemic uncertainty**（模型预测误差导致）和 **aleatoric uncertainty** （标注误差导致） 因此，gt就不能简单的把特征图上gt所在的位置的标签设置成1（类似GFL的思想），而应该是特征图上的每个位置都可能表示第 $j$ 个关键点。所以，关键点的位置在特征图像就是一种分布，而最终预测的关键点位置应该是该分布的期望。本文考虑了两种分布：高斯分布 和 拉普拉斯分布。

### 2.3.1 高斯分布

由于空间位置分布包含 $x, y$ 两个方向，因此位置向量 $[x, y]$ 此时需要服从均值为 $\mu$ ，协方差矩阵为 $\Sigma$ 的二元高斯分布：
$$
f(x, y) = 
\frac{1}{\sqrt{2\pi}\sigma_x} e ^ {- \frac{(x - \mu_x)^2}{2\sigma_x}}
\frac{1}{\sqrt{2\pi}\sigma_y} e ^ {- \frac{(y - \mu_y)^2}{2\sigma_y}}

\\ = 

\frac{1}{2\pi \sigma_x \sigma_y} e ^ 
{- \frac{1}{2} [\frac{(x - \mu_x)^2}{\sigma_x} + \frac{(y - \mu_y)^2}{\sigma_y}]}
$$
令:
$$
f(x, y) = 

\frac{1}{2\pi \sigma_x \sigma_y} e ^ 
{- \frac{1}{2} d}
$$
即：
$$
d = \frac{(x - \mu_x)^2}{\sigma_x} + \frac{(y - \mu_y)^2}{\sigma_y}

\\ =
 
\left[ 
\begin{array}{}
	x - \mu_x, y - \mu_y
\end{array}
\right]

\left[ 
\begin{array}{}
	\frac{1}{\sigma_x} & 0 \\
	0 & \frac{1}{\sigma_y}
\end{array}
\right]

\left[ 
\begin{array}{}
	x - \mu_x \\
	y - \mu_y
\end{array}
\right]

\\ = 

(X - \mu)^T \Sigma^{-1} (X - \mu)
$$
上式协方差矩阵的逆的非对角线元素是0，是由于 $x, y$ 方向上的两个随机变量相互独立（如人脸x不变，y可以任意移动）。

此外 $\sigma_x \sigma_y $ 可以表示为行列式：
$$
\sigma_x \sigma_y 
= 
\sqrt{
\left|
\begin{array}{}
	\sigma_x^2 & 0 \\
	0 & \sigma_y^2
\end{array}
\right|
}
= \sqrt{\sigma_x^2 \sigma_y^2}
$$
因此，二元位置分布函数最终可以表示为：
$$
f(x, y) = 

\frac{exp(-\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu)) }{2\pi \sqrt{|\Sigma|}}
$$
其中, $X$ 是第 $j$ 个特征图上关键点的真值 $p_j$，$\mu$ 为该关键点位置的期望 $\mu_j$，由模型预测得到。$\Sigma_j$ 由关键点的真值和模型预测的期望 $\mu_j$ 计算得到。

## 2.2 Visiability Estimator

可见性分支预测整张图像上每个点的可见性，因此输出维度是一个 $N_p$ 维的特征向量，即对于关键点的每个特征图，只预测出一个可见性的数值，并通过Sigmoid函数来表示成可见性概率 $\hat{v_{j}}$ ，显然，关键点可见性是一个伯努利分布（1可见，0不可见）$q(v_j) = \hat{v_j}^v(1-\hat{v_j})^{(1 - v)}$，参数为 $v_j$ ：

- $v \in \{0, 1\}$

- $q(v = 1) = \hat{v_j} $
- $q(v = 0) = 1 - \hat{v_j} $

由于第 $j$ 个关键点的位置服从 2.3 Location Likelihood 的某种分布 $P$，而该关键点的可见性也服从一种分布 $q(v)$（伯努利分布）。而每个关键点是否可见是依赖于该关键点的空间位置的（如，有些关键点所在的位置被遮挡）,但是之前的方法都只考虑 $p(v)$ ，未考虑位置分布 $P$ 。因此本文的关键点可见性估计使用的是 可见性分布 和 位置分布的联合分布 $q(v, p)$ ：

+ 边缘概率分布 $q(v) = $ 服从伯努利分布，表示关键点是否可见
+ 边缘概率分布 $q(p)$ 服从  2.3 Location Likelihood 中假设的分布，表示关键点的空间位置

每个关键点的可见性和位置分度的联合分布函数可以利用全概率公式表示成 $q(v, p) = q(v) q(z|p)$ ：

- $q(v)$ 是模型预测的可见性概率，参数为 $v$

+ $q(p | v = 1)$ 表示可见关键点的位置分布。参数因此与位置分布 $P$ 相同，为高斯分布或伯努利分布的 $\mu, \Sigma$
+ $q(p | v = 0)$ 表示不可见关键点的位置分布

对于不可见关键点，有两种情况导致不可见：

![image-20230421113506248](imgs/1-LUVLi%20Face%20Alignment%20Estimating%20Landmarks%27%20Location,%20Uncertainty,%20and%20Visibility%20Likelihood/image-20230421113506248.png)

- 自身遮挡 ： 如侧脸。这部分被遮挡的关键点的标注难度较大，且标注误差一般也较大，如上图中黑色的关键点。因此，在计算Loss时，这部分关键点不计算联合分布的似然损失，可见性$ v = 0$。
- 外部遮挡：如眼镜，手，头发。这部分遮挡的关键点的标注难度较小，如上图中红色的关键点。因此，在计算Loss时，这部分关键点的可见性也所作可见 $ v = 1$

因此：

- $q(p | v = 1)$ 表示可见，和外部遮挡的关键点的位置分布（外部遮挡的可见性在计算loss时当成可见）。因此，$q(p | v = 1) = P(\mu_j, \Sigma_j)$
- $q(p | v = 0)$ 表示自身遮挡时，关键点不可见的位置分布，由于自身遮挡的标注误差较大，所以不计算联合分布的似然损失。即，此时$q(p | v = 0) = \emptyset$ ，表示位置分布不存在。

为了从联合概率分布中估计参数 $v, \mu, \Sigma$ ，本文采用的是极大似然估计，第 $j$  个关键点的对数似然为：
$$
L_j = - ln(q(v_j, p_j))
\\= 
-ln(q(v_j) q(p_j|v_j))
\\=
- [ln(q(v_j)) + ln(q(p_j|v_j))]
\\=
- ln(q(v_j)) - ln(q(p_j|v_j))
$$
其中：
$$
ln(q(v_j)) = ln ( \hat{v_j}^{v_{j}}(1-\hat{v_j})^{(1 - {v_{j}})} )
\\=
v_jln(\hat{v_j}) + (1 - v_j)ln(1 - \hat{v_j})
$$

$$
ln(q(p_j|v_j)) = v_j ln(P(\mu_j, \Sigma_j))
$$

其中，$v P(\mu_j, \Sigma_j)$ 中的 $v$ 可以认为是指示函数，对于可见关键点和外部遮挡的关键点，指示函数是1，表示此时计算位置分布的似然函数。而自身遮挡的关键点的指示函数是0，表示此时不计算位置分布的似然损失。

把 2.3 Location Likelihood 中的 $P(\mu_j, \Sigma_j)$ 带入上式：
$$
ln(q(p_j|v_j)) = v_j ln(P(\mu_j, \Sigma_j))
\\=
v_j ln( \frac{exp(-\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu)) }{2\pi \sqrt{|\Sigma|}} )
$$
其中, $X$ 是第 $j$ 个特征图上关键点的真值 $p_j$，是观察到的样本。$\mu$ 为该关键点位置的期望 $\mu_j$，由模型预测得到。$\Sigma_j$ 由关键点的真值和模型预测的期望 $\mu_j$ 计算得到。模型优化的目的是使得观察到的真值位置 $p_j$ 的概率最大，$p_j$ 属于高斯分布或拉普拉斯分布，因此通过极大似然估计分布函数的参数 $\mu_j$ 和 $\Sigma_j$ ：
$$
ln(q(p_j|v_j)) = v_j ln(P(\mu_j, \Sigma_j))
\\=
v_j ln( \frac{exp(-\frac{1}{2} (p_j - \mu_j)^T \Sigma_{j}^{-1} (p_j - \mu_j)) }{2\pi \sqrt{|\Sigma_j|}} )
\\=
v_j ln(exp(-\frac{1}{2} (p_j - \mu_j)^T \Sigma_{j}^{-1} (p_j - \mu_j))) - v_j ln( 2\pi \sqrt{|\Sigma_j|} )
\\=
-v_j \frac{1}{2} (p_j - \mu_j)^T \Sigma_{j}^{-1} (p_j - \mu_j) - v_j ln( 2\pi \sqrt{|\Sigma_j|} )
\\=
-v_j \frac{1}{2} (p_j - \mu_j)^T \Sigma_{j}^{-1} (p_j - \mu_j) - v_j ln( \sqrt{4 \pi^2 |\Sigma_j|} )
\\=
-v_j \frac{1}{2} (p_j - \mu_j)^T \Sigma_{j}^{-1} (p_j - \mu_j) - \frac{1}{2}v_j ln( 4 \pi^2 |\Sigma_j| )
$$
为了计算简单，$ln( 4 \pi^2 |\Sigma_j| )$ 可以等价替换为以某个常数为底的形式：$ln( 4 \pi^2 |\Sigma_j|) = log( |\Sigma_j| )$ 。**因此，最终的损失为：**
$$
L_j = - ln(q(v_j, p_j))
\\= 
- ln(q(v_j)) - ln(q(p_j|v_j))
\\= 
- v_jln(\hat{v_j}) + (1 - v_j)ln(1 - \hat{v_j})
+
v_j \frac{1}{2} (p_j - \mu_j)^T \Sigma_{j}^{-1} (p_j - \mu_j) + \frac{1}{2}v_j ln( 4 \pi^2 |\Sigma_j| )
$$

$$
L = \frac{1}{N_p} \sum_{j=1}^{N_p} L_j
$$



# 3 实验

## 3.1 不确定性预测的准确性

![image-20230421143342853](imgs/1-LUVLi%20Face%20Alignment%20Estimating%20Landmarks%27%20Location,%20Uncertainty,%20and%20Visibility%20Likelihood/image-20230421143342853.png)

上图从左到右分别为两个自协方差，和一个协方差。即，网络输出的3个值。横轴是模型预测结果，纵轴是gt。可以发现，两者的正相关很高。

