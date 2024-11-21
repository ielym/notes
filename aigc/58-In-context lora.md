# 0 引用

- [In-Context LoRA for Diffusion Transformers](https://arxiv.org/abs/2410.23775)

# 1 介绍

- 假设t2i DiT内在就具有上下文生成能力，只需要很少的微调就能激活这种能力。
- 通过多种不同任务的实验，我们从定性上证明了DiT模型可以在不进行任何微调的情况下，有效的进行in-context生成。
- 基于该观察，利用in-context能力提出了一种非常简单的pipeline：
  - 在图像维度concat，而不是token维度
  - 对多张图像组合打标
  - 使用20-100张的特定任务的少量数据，用lora fine-tune模型
  - 模型成为In-Context LoRA (IC-LoRA)

---

![image-20241119002139942](imgs/58-In-context%20lora/image-20241119002139942.png)

关键假设的实验基于上图：

- 使用FLUX.1-dev模型

- 观察到t2i模型已经可以处理不同任务了，尽管有一些瑕疵

- 已经能够保持主题的属性，风格，光照，色彩等一致性，同时也能够修改一些如姿态，角度和布局的属性

- 此外，也能够在同一个prompt中响应不同的描述，上图的caption为：

  ![image-20241119002531466](imgs/58-In-context%20lora/image-20241119002531466.png)

​		![image-20241119002548202](imgs/58-In-context%20lora/image-20241119002548202.png)

---

对于上述发现，可以得到一些关键的insights：

- Inherent In-Context Learning ：t2i模型已经具有上下文生成能力。通过适当地触发和增强这种能力，我们可以利用它来完成复杂的生成任务。

- Model Reusability Without Architectural Modifications ：由于t2i模型可以处理合并的captions，因此可以复用这种能力来做in-context生成。不需要修改模型，只需要修改输入。

- Efficiency with Minimal Data and Computation ：高质量的结果不需要大量数据和长时间的训练。小但高质量的数据，以及少量的计算资源可能已经足够了。

# 2 Method

## 2.1 问题定义

- 定义大多数的图像生成任务为：基于 $m \ge 0$ 个condition图像，以及 $n+m$ 个text prompts，生成 $n \ge 1$ 个图像。
- IC-LoRA使用一个统一的prompt用于全部图像，这种统一的提示设计与现有的文本到图像模型更加兼容。

## 2.2 In-Context LoRA

- 同时生成一组图像，训练时concat成一个大图像。
- 只有一个统一的prompt，包含一个总的description，并合并每张图像的captions
- 生成一张大图之后，切分成多个panels。
- 不微调基模，使用LoRA，在小但高质量的数据上训练。

# 3 实验

## 3.1 Implementation Details

- 使用FLUX.1-dev模型
- 每个任务单独训练
- 任务包含故事生成，字体设计，肖像照片，视觉设计，家装， visual effects，portrait illustration， PowerPoint template design。
- 对于每个人物，搜集20-100个高质量图像。
- 每组图像concat成一个图像，使用MLLMs对这些图像打标;
  - 最开始是一个汇总的prompt，之后是每个图像的详细的描述。
  - 使用单个A100，训练5000steps，bs=4, LoRA rank=16
- 推理时：
  - 采样20步，guidance scale=3.5
- 对于图像条件生成的任务，使用SDEdit

# 4 结果

## 4.1 不需要参考图的生成

### Film Storyboard Generation

![image-20241120004459260](imgs/58-In-context%20lora/image-20241120004459260.png)

### Potrait Photography

![image-20241120004530719](imgs/58-In-context%20lora/image-20241120004530719.png)

### Home Decoration

![image-20241120004550550](imgs/58-In-context%20lora/image-20241120004550550.png)

### Couple Profile Generation

- 情侣头像生成

![image-20241120004610438](imgs/58-In-context%20lora/image-20241120004610438.png)

### Font Design

![image-20241120004643925](imgs/58-In-context%20lora/image-20241120004643925.png)

### PPT Template Design

![image-20241120004704036](imgs/58-In-context%20lora/image-20241120004704036.png)

### Visual Identity Design

![image-20241120004722468](imgs/58-In-context%20lora/image-20241120004722468.png)

### Standstorm Visual Effect

![image-20241120004752797](imgs/58-In-context%20lora/image-20241120004752797.png)

### Portrait Illustration

![image-20241120004811595](imgs/58-In-context%20lora/image-20241120004811595.png)

## 4.2 参考图

- 使用SDEdit

![image-20241120004903117](imgs/58-In-context%20lora/image-20241120004903117.png)

## 4.3 Failure Cases 

![image-20241120004952229](imgs/58-In-context%20lora/image-20241120004952229.png)
