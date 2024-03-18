`PROMPT-TO-PROMPT IMAGE EDITING WITH CROSS-ATTENTION CONTROL`



# 2 Method

定义：

- $I$ 是text-guided的扩散模型生成的图像，prompt 是 $P$ ，随机种子是 $s$ 

- 我们的目标是使用一个编辑过的prompt $P^*$ 来编辑 $I$ ，得到 $I^*$ 。其中，$I^*$ 即需要保持原始的内容和结构，又需要与编辑后的prompt相关。

与其他方法不同，我们想通过不适用任何用户定义的mask来实现图像编辑。为了达到这一点，最简单的方法是固定随机数种子，修改prompt，然后重新生成，但这样的结果与原始结果差别较大，如下图第二行所示：

![image-20230822144722486](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230822144722486.png)

我们观察到，生成图像的外观和结构不仅依赖于随机数，也依赖于扩散过程中像素和文本编码的交互。通过修改 cross-attention 层中的 pixel-to-text 交互，我们能够实现 prompt-to-prompt的图像编辑能力。

## 2.1 Cross-Attention In Text-Conditioned Diffusion Models

使用 Imagen 作为baseline，同时也在 Latent Diffusion, Stable Diffusion上进行了实验。三者的交叉注意力机制分别为：

### 2.1.1 Imagen

Imagen包含3个 text-conditioned diffusion 模型和1个语言模型：

- 一个 text-to-image 的 64x64 的扩散模型
- 两个超分扩散模型，其中一个用于 64x64 -> 256x256，一个用于 256x256->1024x1024
- 一个预训练的 T5 XL 语言模型

其中，三个扩散模型的差别为：

- 64x64 ：从随机噪声开始，在分辨率为 [32, 16, 8] 的层上使用 cross-attention 和 hybrid-attention 来注入文本编码
- 64x64 -> 256x256 ：在bottleneck（分辨率32）上使用hybrid-attention
- 256x256 -> 1024x1024 ：在 bottleneck （分辨率64）上使用 cross-attention

### 2.1.2 Latent Diffusion

Latent Diffusion与Imagen不同，首先为了减少内存消耗，LDM使用VGGAN输出的隐空间，该过程把分辨率从 256减少至 32 ，通道数为4。

语言模型由 32 个 transformer层组成。

扩散模型在分辨率为 32, 16, 8, 4 时，使用self-attention + text-conditioned cross-attention

### 2.1.3 Stable Diffusion

SD 是 LDM 的修改版，分辨率更高。隐编码是 64x64x4。

语言模型是CLIP。

扩散模型在分辨率为 64, 32, 16, 8 时，使用self-attention + text-conditioned cross-attention

### 2.1.4 Cross-Attention

定义：

- 文本编码 $\psi(P)$ 
- 时间步 $t$ 的噪声图像 $z_t$ ，Unet的预测噪声为 $\epsilon$ 
- 扩散模型最后一步时，$I = z_0$ 
- Unet中，即将进行 Cross-Attention 的空间特征为 $\phi(z_t)$ 

![image-20230822152403416](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230822152403416.png)

如上图所示：

- $Q = l_Q(\phi(z_t))$ 
- $K = l_K (\psi(P))$
- $V = l_V (\psi(P))$ 
- $M = Softmax(\frac{QK^T}{\sqrt{d}})$ 。其中，$M_{ij}$ 表示第 $j$ 个token和第$i$ 个像素的权重，$d$ 是 $K, V$ 的维度。
- 最终的输出特征为 $\hat{\phi}(z_t) = MV$   

## 2.2 Controlling the Cross-Attention

![image-20230822154146168](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230822154146168.png)

像素和文本的交互如上图所示 （$M \in \mathbb{R}^{heads \times hw \times 77}$ ，因此可以画出热力图，画的时候多头注意力的权重求了平均）：

- 从上图第一行：像素与对应描述他的文本的注意力更大。如，像素中熊的区域的注意力机制在 'bear' 的文本时最大
- 从第二行：在去噪早期，物体的结构就已经确定了。

---

具体的编辑流程如下：

定义：

- 时间步 $t$ 的隐变量 $z_t$ 
- Prompt $P$
- 随机数种子 $s$

- $DM(z_t, P, t, s)$ 是单个时间步 $t$ 的扩散过程，输出噪声图像 $z_{t-1}$ ，扩散过程中的attention map是 $M_t$ 
- 在 $DM(z_t, P, t, s)$ 过程中，使用 attention map $\hat{M}$ 替换掉 $M$ 表示为 ：$DM(z_t, P, t, s) \{ M \leftarrow \hat{M} \}$ 
- $M_t^*$ 表示使用 $P^*$ 的 attention map
- 编辑函数为 $Edit(M_t, M_t^*, t)$ 

算法流程如下：

![image-20230823164238048](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230823164238048.png)

- Input ：源prompt $P$ ，目标prompt $P^*$ ，随机数种子 $s$ 
- Optional for local editing ：$w, w^*$ 表示编辑前后的单词，用于指定编辑区域（不同单词在attention map 上热力图的范围不同），因此可以用于局部编辑（把attention map的其他区域固定不变）
- Output ：源图像 $x_{src}$ 和编辑后的图像 $x_{dst}$ 

---

- 使用随机数种子 $s$ 生成 $z_T^* = x_T \in N(0, I)$ ，$z_T^*, z_T$ 完全相同，是赋值得到的。
- for t = T, T-1, ..., 1 do:
  - 使用原始的隐编码，prompt得到 $z_{t-1}, M_t = DM(z_t, P, t, s)$ 
  - 使用编辑的隐编码，prompt得到 $M_t^* = DM(z_t^*, P^*, t, s)$ 
  - 通过 $Edit(M_t, M_t^*, t)$ 函数替换/修改两个attention map的值，得到修改后的 attention map $\hat{M_t}$ 
  - 使用修改后的 $\hat{M_t}$ 替换掉编辑的 $M_t^*$ ，并得到编辑的隐编码 $z_{t-1}^* \leftarrow DM(z_t^*, P^*, t, s)\{ M \leftarrow \hat{M_t} \} $
  - 局部编辑的部分后面介绍 $Edit$ 函数时会讲

其中，$Edit(M_t, M_t^*, t)$ 函数针对不同类型的修改也不同，具体如下：

### 2.2.1 Local Editing

Local Editing 表示修改图像中的某个物体，同时保持其他部分（如背景不变）。此时 Edit 函数如下：

- 估计attention mask 中需要编辑的区域，并保持非该区域的 attention mask 不变。为了计算时间步 t 时的编辑区域的mask ：
  - 取出 $T, T-1, T-2, ..., t$ 的所有的 $M_t$ 和 $M_t^*$ 
  - 计算 $M_t, M_t^* \in \mathbb{R}^{heads \times hw \times 77}$ 中的每个单词 $w, w^* = 1, 2, 3, ... 77$ 的 $M_w, M_w^* \in \mathbb{R}^{heads \times hw}$  ，从 $T, T-1, ..., t$ 的每个 $M_w, M_w^*$ 的均值，得到 $\bar{M}_{t, w}, \bar{M}_{t, w^*}^* \in \mathbb{R}^{heads \times hw}$ 。
  - 其中，$\bar{M}_{t, w}, \bar{M}_{t, w^*}^*$ 上任意位置的激活值 $x \gt k$ 的位置，选择为需要编辑的区域。$k = 0.3$ 。对于改变几何形状的任务，需要同时考虑编辑前和编辑后需要改变的区域，因此，实验中最终的编辑区域掩码 $\alpha$ 是 $\bar{M}_{t, w}, \bar{M}_{t, w^*}^*$ 的并集。

再看算法中：

![image-20230823175447171](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230823175447171.png)

也就是表示：

- 需要编辑的区域用 $z_{t-1}^*$ 的值
- 不需要编辑的区域用  $z_{t-1}$ 的值

### 2.2.2 Word Swap

该情况表示用户想要替换掉原始 prompt 中的一个词，比如 `a big bicycle` 替换为 `a big car` 。

主要问题：汽车 -> 自行车，自行车 -> 汽车，需要改变的区域过大，容易出现部分区域变不过去的问题。

为解决该问题，使用了一种 softer attention constrain 方法：

![image-20230823180250290](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230823180250290.png)

其中，$\tau$ 时时间步参数，用于决定到哪个时间步才开始注入 $M_t^*$ 。

如果替换前后的两个prompt的字符数不一样，解决方法如 2.2.3 所示

### 2.2.3 Prompt Refinement

该情况对应于加入新的prompt，如 : `a castle` 编辑成 `children drawing of a castle` 。

为了保持通用的细节（如，把一幅照片变成一张画），我们只在两个prompt通用的prompt上进行注入：

- 使用一个对齐函数 $A$ ，输入 $P^*$ 的 prompt 索引，并输出 $P^*$ 的prompt的索引在 $P$ 上的索引（如果 $P$ 中没有该prompt，则输出None）

- 编辑过程如下：

  ![image-20230823190834932](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230823190834932.png)

其中，$i$ 表示像素值，$j$ 表示一个text token。

### 2.2.4 Attention Re-weighting

用户可能想增强一个token的概念，比如，在 $P$ = "a fluffy ball" 中，假设用户想控制球的毛茸茸的程度（更强，更弱）。为实现该编辑，我们可以使用缩放因子 $c \in [-2, 2]$ 缩放attention map：

![image-20230913105854278](imgs/29-PROMPT-TO-PROMPT%20IMAGE%20EDITING%20WITH%20CROSS-ATTENTION%20CONTROL/image-20230913105854278.png)

- 给想要编辑的token索引$j^*$ 增大权重，用 $c$ 来控制。
