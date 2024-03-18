# 1 量化算法

为了量化模型，可以使用 `snpe-dlc-quantize` 工具，本节加介绍 SNPE 的量化算法。

需要注意的是，SNPE 提供了多种量化模式，本节将详细介绍默认的量化模式。

+ 计算输入图像的最大值和最小值：
  + 如，输入数据为 $[-1.8, -1.0, 0, 0.5]$
  + $true\_min = -1.8$ ，$true\_max = 0.5$
  
+ 计算 $encoding\_min$ 和 $encoding\_max$ 。其中，encoding-min表示量化后的最小值将会表示成0，encoding-max表示量化后的最大值将会表示成255。

  + 首先，把 $encoding\_min$ 和 $encoding\_max$ 初始化为 $true\_min = -1.8$ ，$true\_max = 0.5$

  + 如果把  $encoding\_min$ 和 $encoding\_max$ 分成255份，则每份的步长是 ：
    $$
    step\_size = \frac{encoding\_max - encoding\_min}{255} = \frac{2.3}{255} = 0.009020
    $$
    



```python
import numpy as np

input_data = np.array([-1.8, -1.0, 0, 0.5], dtype=np.float32)

# 实际的最大值和最小值
true_min = np.min(input_data)
true_max = np.max(input_data)

# 按照实际的最大、最小值，划分成255份，每份有多少
range_between_max_min = true_max - true_min
step_size = range_between_max_min / 255. # 0.00901960765614229

# 需要保证 0 能够准确的用整数表示
# 此时 0 被表示成多少了？
rest_zero = abs(true_min) % step_size # 0.005098028743968376
exactly = False
if abs(rest_zero - 0.0) < 1e-7: # 设置精度
    exactly = True

# 如果 0 不能被表示成整数，移动 true_max, true_min。使 0 能够被准确表示
if not exactly:
    # 余数占 step_size 的多少，从而判断此时 0 是离左侧的整数近，还是离右侧的整数近
    if rest_zero / step_size > 0.5:
        # 离右侧的整数仅，为了保证：1. 0 被准确的用整数表示；2. 移动尽可能少的偏移。因此需要整体左移
        # 由于此时 true_min < 0, true_max > 0
        # 因此 ：abs(true_min - bias) / step_size = (-true_min + bias) / step_size = round(abs(true_min) / step_size)
        bias = round(abs(true_min) / step_size) * step_size + true_min
        # 偏移后，最大值，最小值分别为
        encoding_min = true_min - bias
        encoding_max = true_max - bias
    else:
        # 同理，整体需要右移
        # 由于此时 true_min < 0, true_max > 0
        # 因此 ：abs(true_min + bias) / step_size = (-true_min - bias) / step_size = round(abs(true_min) / step_size)
        bias = - (round(abs(true_min) / step_size) * step_size + true_min)
        encoding_min = true_min + bias
        encoding_max = true_max + bias

# 此时，令 0 表示 encoding_min， 255 表示 encoding_max
# 此时，float 的 0 表示成了：200
float_zero_in_int8 = round(255 * (0 - encoding_min) / range_between_max_min)
# 任意一个浮点数 x 被表示成了
x = 0.1234 # 214
float_x_in_int8 = round(255 * (x - encoding_min) / range_between_max_min) 
```

