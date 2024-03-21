# [Face2Diffusion for Fast and Editable Face Personalization](https://arxiv.org/pdf/2403.05094.pdf)

- 认为浅层特征包含过多和身份不相关的低级信息（camera pose, expression, and background），导致模型依赖于重建任务来最小化MSE损失，编辑性较差。

- 为解决该问题，重新训练了一个特征提取器MSID，MSID即使在浅层也能够判别不同身份的人脸（避免引入身份不相关的低级语义信息），相较于ArcFace的判别程度明显提高：

  ![PiflLW86diRm8jZR](imgs/diffusion_subject_consistency_001/PiflLW86diRm8jZR.png)

# [An Item is Worth a Prompt: Versatile Image Editing with Disentangled Control](https://arxiv.org/abs/2403.04880)

## 1 介绍
![image-20240319005500008](imgs/diffusion_subject_consistency_001/image-20240319005500008.png)在prompt space进行图像编辑更灵活，但由于LDM是用描述性的文本进行训练的，直接修改prompt会导致生成图像的较大变化。此外，大多数编辑方法通过mask来保证非编辑区域不变，这种方法会让DPM忽略对这些区域的生成，导致最终编辑的结果不协调。

## 2 方法

### 2.1 Item-Prompt Association

![image-20240319005520562](imgs/diffusion_subject_consistency_001/image-20240319005520562.png)

定义：把图像 $I$ 分割成 $N$ 个non-overlapped的 items $\{ I_i \}_{i=1}^{N}$ ，对应的 latent $z_i^t$。每个item的 prompt 为$\{ P_i \}_{i=1}^{N}$ ，对应的text embedding为 $c_i = CLIP(P_i)$ 。如Fig. 2所示，cross-attn被修改成分组计算的形式，其中每组的cross-attn可以表示为：

![image-20240319005709848](imgs/diffusion_subject_consistency_001/image-20240319005709848.png)

![image-20240319010154234](imgs/diffusion_subject_consistency_001/image-20240319010154234.png)



### 2.2 Linking Prompt to Item

![image-20240319010218103](imgs/diffusion_subject_consistency_001/image-20240319010218103.png)

如上图所示，对于每个item都使用2个tokens进行表示，并使用服从当前词表分布的均值和方差进行高斯分布初始化。第一个阶段优化随机初始化的tokens，冻结网络：

![image-20240319010239884](imgs/diffusion_subject_consistency_001/image-20240319010239884.png)

其中，$e \in \mathbb{R}^{NM \times D_{emb}}$ ，表示 N 个items，每个item的token维度是 M x D。第二个阶段的重建能力不足，第二个阶段使用相同的损失函数来优化UNet参数。实验发现，只优化cross-attn层相关的参数就足够了。如果有参考图，即一张原图，一张参考图，每张图都有一些items，第二个阶段训练时需要两张一起训练。

### 2.3 Editing with Item-Prompt Freestyle

支持4种图像编辑操作：

![image-20240319010353562](imgs/diffusion_subject_consistency_001/image-20240319010353562.png)

- Text-based Item Editing：直接把一个item的token替换成另一个item的token和文本描述即可：![image-20240319010416269](imgs/diffusion_subject_consistency_001/image-20240319010416269.png)![image-20240319010429084](imgs/diffusion_subject_consistency_001/image-20240319010429084.png)

- Image-based Item Editing：两张图像，用参考图中的item替换source图中的item。

  ![img](imgs/diffusion_subject_consistency_001/image-20240319010525430.png)

- Mask-based Item Editing: 保持item不变，修改segmentation mask（move，resize, refining, redrawing）![image-20240319010618389](imgs/diffusion_subject_consistency_001/image-20240319010618389.png)![image-20240319010633047](imgs/diffusion_subject_consistency_001/image-20240319010633047.png) 

- Item Removal：删除item，缺失的segmentation region用相邻的mask和其对应的item进行补全。 ![image-20240319010633047](imgs/diffusion_subject_consistency_001/image-20240319010633047.png)

# [EMO: Emote Portrait Alive -- Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions](https://arxiv.org/abs/2402.17485)

![image-20240319010825935](imgs/diffusion_subject_consistency_001/image-20240319010825935.png)

# [Magic-Me: Identity-Specific Video Customized Diffusion](https://arxiv.org/abs/2402.09368)

无关背景会影响图文一致性，因此设计一个ID Module（Prompt-to-segmentation）来分割主体。训练时，使用GPT-4V描述主要物体和对应的COCO标签，并输入到Grounding DINO中来获得检测框，之后送到SAM中生成mask。训练时，只计算mask区域的生成loss（没有训练Prompt-to-segmentation，只是为了在训练时不计算背景的生成loss）。![image-20240319010919162](imgs/diffusion_subject_consistency_001/image-20240319010919162.png)效果对比：![image-20240319010940414](imgs/diffusion_subject_consistency_001/image-20240319010940414.png)

# [KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing](https://arxiv.org/abs/2309.16608)

## 1 介绍

![image-20240319011034599](imgs/diffusion_subject_consistency_001/image-20240319011034599.png)

- 主要为了解决现有方法难以编辑动作的问题（同时保持纹理），提出了 KV Inversion。

## 2 方法 
包含两个阶段：(1) Inversion Stage, (2) Tuning Stage：

- Inversion Stage用来生成 $z_t$，使用DDIM Inverse。

- Tuning Stage用来学习 KV Embedding来更好的保持内容一致，有两个主要发现：

  - 内容（纹理和身份）主要受self-attn的控制。因此选择学习self-attn的 K, V来保持内容一致性。
  - 结构（物体位置）主要受 cross-attn的 attention map决定![image-20240319011236437](imgs/diffusion_subject_consistency_001/image-20240319011236437.png)![image-20240319011247726](imgs/diffusion_subject_consistency_001/image-20240319011247726.png)其中，$(K_{l,t}, V_{l,t}) , (\bar{K}_{l,t}, \bar{V}_{l,t})$ 分别表示原始的K, V和新的K, V，按权重加权。
  - 这里 K, V只是一个和原始网络的 $w_k x, w_v x$ 的输出维度相同的embedding，而并不是新加的网络层。为了优化embedding：
    - 用DDIM Inverse第 $t$ 步的噪声作为输出
    - 用source prompt作为文本，预测出 $t-1$ 步的噪声
    - 用DDIM Inverse第 $t-1$ 步的噪声作为gt，计算MSE损失推理时，把优化后的 KV embedding添加到去噪过程中。

- 推理时，把优化后的 KV embedding添加到去噪过程中。

  ![image-20240319011522616](imgs/diffusion_subject_consistency_001/image-20240319011522616.png)



# [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2108.02938)

![image-20240319011656793](imgs/diffusion_subject_consistency_001/image-20240319011656793.png)

- 提出 Iterative Latent Variable Refinement (ILVR) ，根据参考图像生成语义一致的图像。

## 2 方法
定义：

- $\phi_N (\cdot)$ 表示低通滤波操作，包含N个上采样和下采样，低通滤波后的图像维度保持不变。

- $c$ 为condition，保证生成的图像 $\phi_N (x_0) = \phi_N (y)$。

### 2.1 Iterative Latent Variable Refinement
- 在latent space中，有 $x_t，y_t$。

- 生成 $x_{t-1}$ 时，需要保证 $\phi_N (x_{t-1}) = \phi_N (y_{t-1})$ 。而实际预测的 t-1 时刻的图像为 $x_{t-1}^{'}$，因此需要对预测输出进行 refine，整体算法为：

![image-20240319012213625](imgs/diffusion_subject_consistency_001/image-20240319012213625.png)

![image-20240319012248579](imgs/diffusion_subject_consistency_001/image-20240319012248579.png)

# [MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://arxiv.org/abs/2304.08465)

## 1 介绍

![image-20240320015428784](imgs/diffusion_subject_consistency_001/image-20240320015428784.png)

![image-20240319012804380](imgs/diffusion_subject_consistency_001/image-20240319012804380.png)

- 提出MasaCtrl，tuning-free的方法实现复杂的non-rigid图像一致性编辑。
- 可针对生成图像和真实图像进行编辑。

## 2 方法

![image-20240319015735532](imgs/diffusion_subject_consistency_001/image-20240319015735532.png)

定义：原图像 $I_s$ , 源文本 $P_s$ ，目标图像 $I$ ，目标文本 $P$ 。

- 获得原图像，目标图像的初始噪声：

  - 对source真实图像，把文本设置成空并使用DDIM获得初始噪声 $z_T^s$ ；
  - 对source是生成图像时，直接使用生成时的噪声作为 $z_T^s$；
  - 对目标图像，随机生成初始噪声 $z_T$ （或者使用Inverse的初始噪声？论文好像没说）。

- 去噪过程中：

  - source图像用源文本去噪，并获得每一步的 $Q_s, K_s, V_s$ 
  - target图像用目标文本去噪，并获得每一步的 $Q, K, V$ 。
  - 根据两组 $Q, K, V$ 进行编辑，即算法中的 **EDIT** 。

- **EDIT：**

  - 用 $K_s, V_s$   替换 $K, V$ 来保证编辑后的图像与源图像的内容一致。

    - 但是直接这样做会导致最终生成的图像与原图基本相同，无法达到编辑的目的。

    - 因此，基于特征可视化发现：

      - 去噪早期主要形成物体的结构和形状，所以在早期注入source特征会破坏目标图像的布局信息。
      - 进一步观察U-Net浅层（例如编码器部分）中的Query特征无法获得清晰的布局以及修改后的提示对应的结构，因此无法获得具有所需空间布局的图像。

      ![image-20240320003615351](imgs/diffusion_subject_consistency_001/image-20240320003615351.png)

  - 所以，mutual self-attention control只在去噪一定阶段之后，并在decoder中进行注入：

    ![image-20240320014038104](imgs/diffusion_subject_consistency_001/image-20240320014038104.png)

  - mutual self-attenton的结构如下所示：

    ![image-20240319233703056](imgs/diffusion_subject_consistency_001/image-20240319233703056.png)

- 此外，为了避免object区域受背景的影响，还在 mutual self-attention中加入了mask：

  - 在时间步 $t$ 时，源图像和目标图像分别执行一次前向传播，来获得cross-attention的 attention score。

    ![image-20240320015129814](imgs/diffusion_subject_consistency_001/image-20240320015129814.png)

  - 之后对所有 $16 \times 16$ 的层和所有head的cross-attention maps进行平均，得到 $A_t^c \in \mathbb{R}^{16 \times 16 \times N}$ ，$N$ 是tokens数量。根据前景token，从源图像和目标图像中获取的attention map分别未 $M_s$ 和 $M$ 。

  - 因此，就可以限制目标 $Q$ 仅从源图像的object区域query前景，并且 $Q$ 只在源图像的背景区域query背景，之后两部分组合起来：

    ![image-20240320014946495](imgs/diffusion_subject_consistency_001/image-20240320014946495.png)

- 该方法也能够和 controlNet或T2I-Adapter等结合使用：

  ![image-20240320015630599](imgs/diffusion_subject_consistency_001/image-20240320015630599.png)



# [MagicMix: Semantic Mixing with Diffusion Models](https://arxiv.org/abs/2210.16056)

![image-20240320015848636](imgs/diffusion_subject_consistency_001/image-20240320015848636.png)

## 1 介绍

- 任务是融合多个概念，如柯基和榨汁机。

- 可进行 image-text mixing 和 text-text mixing 两个任务。

## 2 方法

![image-20240322005227965](imgs/diffusion_subject_consistency_001/image-20240322005227965.png)

- **Image-txt mixing ：**
  - 对原图DDIM Inverse加噪，不需要加到 $x_T$ ，只需要加到合适的 $x_{K_{max}}$ 即可。
  - 在加噪的过程中，可以得到一系列的中间噪声 $\{ x_k \}_{K_{min}}^{K_{max}}$，$K_{min}$ 不一定是$x_0$ ，是某个中间时间步 。不同噪声都包含不同的布局信息和轮廓信息，由粗到细。
  - 之后开始去噪，去噪时使用文本 $y$ 作为 condition，去噪的初始噪声为 $\hat{x}_{K_{max}} = x_{K_{max}}$ 。去噪时的每一步噪声为 $\hat{x}^{'}_{k-1}$ ，对应的加噪时的噪声为 $x_{k-1}$ 。两个噪声使用一个固定的线性加权权重 $v \in [0, 1]$ ，得到最终的噪声 $\hat{x}_{k-1} = v \hat{x}^{'}_{k-1} + (1 - v) x_{k-1}$ 。
  - 去噪到 $K_{min}$ 时间步之后，从 $K_{min}$ 到 $t=0$ 中，就不再引入加噪时的噪声了，只使用去噪的噪声。

- **Text-text mixing :** 

  - 与文本图像融合类似，只不过image-text mixing的加噪噪声此时替换成了 $p_\theta (x_k | y_{layout})$ 。即，先用一个文本生成图像，记录中间噪声。
  - 之后使用另一个文本 $y_{content}$ 进行去噪，然后在 $K_{max} \to K_{min}$ 时间步中进行融合。

- **Weighted Image-Text Cross-Attention :** 

  - 上述两个应用都有去噪文本，可以利用P2P的 reweighting的方式对某些token进行缩放，如：

    ![image-20240322011618386](imgs/diffusion_subject_consistency_001/image-20240322011618386.png)

  - s 可以是正数也可以是负数，负数是能够移除某些概念。



# [IDAdapter: Learning Mixed Features for Tuning-Free Personalization of Text-to-Image Models](https://arxiv.org/abs/2403.13535)

![image-20240321234152760](imgs/diffusion_subject_consistency_001/image-20240321234152760.png)



## 1 介绍

ID保持的要求：（1） 生成图像的风格需要匹配prompt，（2）生成图像的角度需要包含多种姿态，（3）能够生成不同表情的图像。

然而现有方法通过在 textual space embedding 注入特征的方式对ID特征的表达能力受限，因此需要用基于图像特征的引导来补充 textual condition。

我们发现，无论是使用CLIP图像编码还使用人脸识别网络，都会绑定非id信息（如，生成图像中的表情，风格，姿态等都绑定到了人脸上，灵活性受限）。为解决该问题，提出了 Mixed Facial Features module (MFF), 用于控制解耦 ID 和 non-ID 特征。

## 2 方法

- 训练时输入 N 张同一个人的人脸图像 $x^{(1)}, x^{(2)}, ..., x^{(N)} $ ，其中 $x^{(i)} \in \mathbb{R}^{h \times w \times c}$ 。如果不够 $N$ 张图像，则通过随机翻转，旋转，颜色变换等各种数据增强来扩充到 $N$ 张。

- 使用 CLIP vision model 得到：

  -  patch feature $f_v^{(1)}, f_v^{(2)}, ..., f_v^{(N)}$ ，其中，$f_v^{(i)} \in \mathbb{R}^{p^2 \times d_v}$ 
  -  cls embeddings  $f_c^{(1)}, f_c^{(2)}, ..., f_c^{(N)}$ ，其中 , $f_c^{(i)} \in \mathbb{R}^{1 \times d_v}$ 
  - $N$ 张图像的 patch feature concat起来，得到 $f_v = Concat (\{ f_v^{(i)} \}_1^N)$ 。该特征来自于具有相同id的多个图像，因此会增强共有的身份信息，并会弱化其他如角度，表情等信息。实验发现，$N = 4$ 对一致性和编辑性最好。

- 使用 Arcface得到：

  - 身份特征 $\{ f_a^{(1)}, f_a^{(2)}, ..., f_a^{(N)} \}$ ，其中，$f_a \in \mathbb{R}^{1 \times d_a}$ 。
  - 之后计算均值得到 $f_a = \sum_{i=1}^N f_a^{(i)} / N$ 

- 之后 $f_a， f_v$  一起送到一个轻量级的 transformer $P_{visual}$ （MFF） 中得到 $E_r = P_{visual} ([f_v, f_a])$ ，如下图所示：

  ![image-20240321235914094](imgs/diffusion_subject_consistency_001/image-20240321235914094.png)

- 文本注入：

  - 在prompt最后加上一个 $sks$ ，需要注意，$sks$ 只是为了在进行tokenizer时作为一个占位符，便于索引，并不是为了更新 $sks$ 的向量。
  - 对于 CLS编码 $f_c$ ，计算均值 $E_c = P_{textual (\sum_{i=1}^N f_c^{(i)} / N) }$ ，$P_{textual}$ 是一个MLP，为了获得紧凑的概念。
  - 之后，根据 sks 的索引，在 text encoder的embedding layer用$E_c$替换sks。
  - 正常使用cross-attention进行注入，并打开cross-attn的 $W_k, W_v$ 。

- 图像注入

  - 额外加入了一个adapter layer：
    $$
    y = y + \beta tanh(\gamma) S([y, E_r])
    $$

  - 其中，$y$ 是 self-attn的输出，$S$ 是adapter layer的self-attn，$\gamma$ 初始化成0。

- 损失计算：

  - Face Identity Loss:
    $$
    L_{id} = E[1 - cos(R(\hat{x_0}), f_a)]
    $$

  - $\hat{x_0}$ 是根据 $z_t$ 计算的，可能不准确，因此加了一个门控，只有当人脸检测模型 $F(\hat{x_0})$ 能检测出来人脸时才会计算，此时 $F(\hat{x_0}) = 1$ ，否则为0 。

  - 最终的 loss 为：
    $$
    L = L_{SD} + F(\hat{x}_0) \lambda L_{id}
    $$

- 训练策略：

  - ~50000 iters
  - $\lambda = 0.1$ 
  - lr = 3e-5
  - bs = 4
  - 1 A100

- 推理时仅重复一张图像 $N$ 次。



# [OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](https://arxiv.org/abs/2403.10983)

