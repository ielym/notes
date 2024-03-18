# 1 介绍

使用 Dreambooth 可以微调SD模型，即可实现特定风格的训练，也可以实现通用大模型的微调。Dreambooth 训练分为两种：

- 基于 LoRA 来 finetune unet 和 text_encoder，实现特定人物，特定场景的微调
- 直接 finetune  unet 和 text_encoder，实现大模型的微调

# 2 Lora Rank

## 2.1 非特定人物

实验使用小红书爬取的真人数据，使用 `wd14-vit-v2` + `BLIP` 共同打prompt标签。Instance prompt 为 `realistic` ，无 class prompt，无正则化图像。学习率设置为 `1e-5` ，训练 `30000` 个iters 进行对比。除`LoRA Rank` 外，其他配置均相同。

- 第1行 ：SD1.5
-  第2行 `rank=4` 
- 第3行 `rank=8` 
- 第4行 `rank=128`

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688288588257-64.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288597771-67.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288608545-70.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288624479-73.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288637545-76.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288670009-82.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288683865-85.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688288031122-1.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288055322-4.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288075100-7.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288106016-10.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288135125-13.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288157261-16.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288174379-19.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688288252021-22.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288266906-25.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288280744-28.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288294327-31.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288308855-34.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288335721-37.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288349915-40.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688288410262-43.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288958527-91.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288971856-94.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288986307-97.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288999676-100.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289011798-103.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289025000-106.jpeg" width="150"/>
</center>

可以看出：

- 不同的 `rank` 在训练 `30000 iters` 时，均未达到预期效果（小红书中国人图像）。 

- 微调效果随 `Rank` 的增大而逐渐变好，`rank=128` 主观上优于其他实验。

# 3 提示词

## 3.1 非特定人物

实验使用小红书爬取的真人数据，Instance prompt 为 `realistic` ，无 class prompt， `LoRA Rank = 128`，无正则化图像。学习率设置为 `1e-5` ，训练 `30000` 个iters 进行对比。除提示词 不同之外，其他配置均相同。

- 第1行 ：为SD1.5的结果
- 第2行 ：仅使用 `wd14-vit-v2` 
- 第3行 ：仅使用 `BLIP` 
- 第4行 ：使用 `wd14-vit-v2` + `BLIP` 

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688288588257-64.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288597771-67.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288608545-70.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288624479-73.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288637545-76.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288670009-82.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688288683865-85.jpeg" width="150"/>
</center>
<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688289670138-109.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289681941-112.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289694638-115.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289708009-118.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289724810-121.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289732962-124.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289741218-127.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688338471929-403.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338480279-406.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338487288-409.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338495676-412.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338502779-415.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338508769-418.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338517557-421.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688289775478-130.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289791417-133.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289800424-136.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289809453-139.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289815818-142.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289823253-145.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688289832766-148.jpeg" width="150"/>
</center>
<center class="half">
    <img src="" width="150"/>
    <img src="" width="150"/>
    <img src="" width="150"/>
    <img src="" width="150"/>
    <img src="" width="150"/>
    <img src="" width="150"/>
    <img src="" width="150"/>
</center>

可以看出：

- 同时使用 `wd14-vit-v2` + `BLIP` 打prompt标签的效果更好
- 主观上，使用 `wd14-vit-v2` 的效果优于 `BLIP` 

# 3 LoRA && 全量

Dreambooth 包含两种训练方式，一种是基于 LoRA 微调 UNet 和 TextEncoder；另一种是直接微调 UNet 和  TextEncoder 的全部参数。后者需要更大的显存和计算量

## 3.1 非特定人物

实验使用小红书爬取的真人数据，不使用 `instance， class` 提示词， `LoRA Rank = 8`，无正则化图像。学习率设置为 `1e-5` ，训练 `30000` 个iters 进行对比。

- 原始SD1.5 **（第1行）**
- `LoRA Rank = 8` 微调 Unet 和 TextEncoder  **（第2行）**
- 微调 Unet 全部参数 ($768 \times 512$) **（第3行）**
- 微调 Unet 全部参数 ($512 \times 512$) **（第4行）**
- 微调 Unet 和 TextEncoder 的全部参数**（第5行）**

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688290382835-151.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290390296-154.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290396535-157.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290405217-160.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290412353-163.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290419059-166.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290425658-169.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688290776092-172.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290785866-175.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290795070-178.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290807400-181.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290816177-184.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290825104-187.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688290834264-190.jpeg" width="150"/>
</center>
<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688313855505-382.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313864111-385.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313870637-388.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313878040-391.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313886763-394.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313894856-397.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313901695-400.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688340224339-472.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340232466-475.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340240810-478.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340248732-481.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340256964-484.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340264343-487.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340271640-490.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688339552214-445.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688339560550-448.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688339567588-451.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688339577114-454.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688339606837-463.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688339613448-466.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688339624507-469.jpeg" width="150"/>
</center>

- 主观上，微调Unet的效果最好。
- 微调 Unet 和 TextEncoder 的全部参数 容易导致灾难性遗忘（但是该实验没有使用正则化数据，需要再补充实验）

# 4 触发词

## 4.1 非特定人物

实验使用小红书爬取的真人数据，`LoRA Rank = 8`，无 `class prompt` ，无正则化图像，`wd14-vit-v2` + `BLIP` 打 prompt标签。学习率设置为 `1e-5` ，训练 `30000` 个iters 进行对比。

- 第一行 ：SD1.5 + `realistic`

- 第二行 ：SD1.5 + `mi_realistic`

- 第三行 ：SD1.5 + `realistic-chinese`

- 第四行 ：SD1.5 + 不使用 `instance`

  

- 第五行 ：微调 `realistic`

- 第六行 ：微调 + `mi_realistic`

- 第七行 ：微调 + `realistic-chinese`

- 第八行 ：微调 + 不使用 `instance`

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688291723215-193.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291731981-196.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291741605-199.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291753261-202.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291759331-205.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291765011-208.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291771070-211.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688291847479-214.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291854386-217.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291860882-220.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291870011-223.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291878056-226.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291900384-229.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688291908358-232.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688294560262-319.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688294569829-322.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688294577221-325.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688294586150-328.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688294592490-331.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688294598704-334.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688294608187-337.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688292305490-298.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292316300-301.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292322105-304.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292329930-307.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292336106-310.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292343069-313.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292350510-316.jpeg" width="150"/>
</center>



---



<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688291995334-235.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292003685-238.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292010110-241.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292018856-244.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292026303-247.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292034749-250.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292040578-253.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688292056598-256.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292064419-259.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292071620-262.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292080959-265.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292087630-268.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292094278-271.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292100193-274.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688338892718-424.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338900840-427.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338907087-430.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338916166-433.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338926680-436.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338935074-439.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688338942443-442.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688292228429-277.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292235392-280.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292241747-283.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292249646-286.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292256114-289.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292264322-292.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688292271684-295.jpeg" width="150"/>
</center>

- 在 SD1.5 中，不同触发词的效果差别不大
- 微调模型中，不同触发词的效果差别也不大。

# 5 分辨率

## 5.1 非特定人物

实验使用小红书爬取的真人数据，全量微调 Unet，无 `instance, class` ，无正则化图像，`wd14-vit-v2` + `BLIP` 打 prompt标签。学习率设置为 `1e-5` ，训练 `30000` 个iters 进行对比。

### 实验一

推理和训练使用相同分辨率

- 第一行 ：SD1.5 + $512 \times 512$
- 第二行 ：SD1.5 + $768 \times 512$
- 第三行 ：微调 + $512 \times 512$
- 第四行 ：微调 + $768 \times 512$

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688313638571-340.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313651547-343.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313660508-346.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313667104-349.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313675541-352.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313690378-355.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313699673-358.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688313729393-361.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313742169-364.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313774970-367.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313783761-370.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313789948-373.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313798112-376.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313806273-379.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688340224339-472.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340232466-475.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340240810-478.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340248732-481.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340256964-484.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340264343-487.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688340271640-490.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688313855505-382.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313864111-385.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313870637-388.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313878040-391.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313886763-394.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313894856-397.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313901695-400.jpeg" width="150"/>
</center>

- 主观上，使用 $512 \times 512$ 训练的效果优于 $768 \times 512$ 训练

### 实验二

推理固定使用 $768 \times 512$ 

- 第1行 ：训练 $768 \times 512$ ，推理 $768 \times 512$
- 第2行 ：训练 $512 \times 512$ ，推理 $768 \times 512$

<center class="half">
    <img src="imgs/0-finetune-dreamlike/1688313855505-382.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313864111-385.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313870637-388.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313878040-391.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313886763-394.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313894856-397.jpeg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/1688313901695-400.jpeg" width="150"/>
</center>

<center class="half">
    <img src="imgs/0-finetune-dreamlike/test_lora_0_0.jpg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/test_lora_1_0.jpg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/test_lora_2_0.jpg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/test_lora_3_0.jpg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/test_lora_4_0-1688340849502-498.jpg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/test_lora_5_0.jpg" width="150"/>
    <img src="imgs/0-finetune-dreamlike/test_lora_6_0.jpg" width="150"/>
</center>

- 主观上，训练和推理使用相同分辨率的效果最好

# 6 数据量

## 5.1 非特定人物

### exp002 - log4 - 07

- dev - fab57711ef844e121e17e103cc7f1d0a56106e4d

- 微调unet全部参数，`instance=mi_realistic` ，无 `class` 。无正则化图像，无负样本。`wd14-vit-v2` + `BLIP` 打 prompt标签。学习率设置为 `1e-6` 。训练分辨率  $512 \times 512$ 。

- 数据来自于小红书和微博

### exp005 - log2 - 04

- main - db88893c0f7fe185a8bbe625019ed94b37bbfa6a

- 微调unet全部参数，`instance=mi_realistic` ，无 `class` 。无正则化图像，无负样本。`wd14-vit-v2` + `BLIP` 打 prompt标签。学习率设置为 `1e-5` 。训练分辨率按hw_ratio排序，过滤人脸占比小于0.08的图像。

- 数据来自于小红书和微博



人物：

锁定：背景，风格

不锁定：关于人脸的描述，关于发型的描述（如果需要生成对应发型的人）









