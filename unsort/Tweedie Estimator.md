在通过 score-based 方法解释 Diffusion 时，遇到了 Tweedie's Estimator 概念。Tweedie's Estimator 是一种参数估计方法，具体推导如下：



在贝叶斯定理中：
$$
p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)}
$$
为了计算$p(\theta|x)$ ，需要知道似然函数 $p(x|\theta)$ 和 $p(\theta)$

Tweedie's Estimator 是针对高斯分布的，即假设了:
$$
p(x|\mu) \sim N(\mu, \sigma^2)
$$
且假设方差 $\sigma^2$ 已知。

此时，在已有观测样本 $x$ 的条件下，我们可以计算 $p(x|\mu)$ 的条件期望 $E[\mu|x]$ ：
$$
E[\mu|x] = \int \mu P(\mu|x) d\theta
\\=
\int \mu \frac{p(\mu)p(x|\mu)}{p(x)} d\mu
\\=
\int  \frac{\mu p(\mu)p(x|\mu) d\mu}{p(x)}
\\=
\int  \frac{\mu p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} d\mu}{p(x)}
$$
Tweedie's Estimator 中对 $\mu$ 进行了补项：
$$
\mu = x + \sigma^2 \frac{\mu - x}{\sigma^2}
$$
带入上式中，得到：
$$
E[\mu|x] =

\int  \frac{\mu p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} d\mu}{p(x)}

\\=

\int  \frac{(x + \sigma^2 \frac{\mu - x}{\sigma^2}) p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} d\mu}{p(x)}

\\=

\int  \frac{
[
x p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}  
+
\sigma^2 \frac{\mu - x}{\sigma^2} p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} 
]
d\mu}
{p(x)}


\\=

\frac{

\int x p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}
d\mu
+
\int \sigma^2 \frac{\mu - x}{\sigma^2} p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} 

d\mu}
{p(x)}
$$
其中，
$$
\frac{\mu - x}{\sigma^2} \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} 

=

\frac{d[\frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}  ]}{dx}

\\=

\frac{dp(x|\theta)}{dx}
$$
带入得到：
$$
E[\mu|x] =

\frac{

\int x p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}
d\mu
+
\int \sigma^2 \frac{\mu - x}{\sigma^2} p(\mu) \frac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x - \mu)^2}{2 \sigma^2}} 

d\mu}
{p(x)}

\\=

\frac{

\int x p(\mu) p(x|\mu) d\mu
+
\int \sigma^2 p(\mu) \frac{dp(x|\mu)}{dx}

d\mu}
{p(x)}

\\=

\frac{

x \int  p(\mu) p(x|\mu) d\mu
+
\sigma^2 \frac{d}{dx} \int p(\mu) p(x|\mu)

d\mu}
{p(x)}

\\=

\frac{

x p(x)
+
\sigma^2 \frac{dp(x)}{dx}

}
{p(x)}

\\=

x
+
\sigma^2 \frac{\frac{dp(x)}{dx}}{p(x)}
$$
由于
$$
\frac{d \quad log(p(x))}{d \quad x} = \frac{1}{p(x)} p(x)' = \frac{\frac{dp(x)}{dx}}{p(x)}
$$
带入得：
$$
E[\mu|x] =
x
+
\sigma^2\frac{d }{dx} log(p(x))
\\=
x
+
\sigma^2 \nabla_x log(p(x))
$$
