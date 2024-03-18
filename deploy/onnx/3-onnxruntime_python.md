文档： [Get started with ORT for Python](https://onnxruntime.ai/docs/get-started/with-python.html#get-started-with-ort-for-python)

Installation Matrix : https://onnxruntime.ai/

# 1 安装

## 1.1 ONNX Runtime

+ CPU 版本：

  ```bash
  pip install onnxruntime
  ```

+ GPU 版本：

  ```bash
  pip install onnxruntime-gpu
  ```

## 1.2 深度学习框架导出ONNX

+ Pytorch

  ```bash
  ## ONNX is built into PyTorch
  pip install torch
  ```

+ Tensorflow

  ```bash
  pip install tf2onnx
  ```

+ Sklearn

  ```bash
  pip install skl2onnx
  ```



# 2 Pytorch, TensorFlow, Scikit Learn 示例

此处不详细介绍细节，但会在其他对应的笔记中单独详细记录。

参考文档：[Quickstart Examples for PyTorch, TensorFlow, and SciKit Learn](https://onnxruntime.ai/docs/get-started/with-python.html#quickstart-examples-for-pytorch-tensorflow-and-scikit-learn)

以 Pytorch CV 为例：

+ 导入模型

  ```python
  torch.onnx.export(model,                                # model being run
                    torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                    "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                    input_names = ['input'],              # the model's input names
                    output_names = ['output'])            # the model's output names
  ```

+ 检查模型

  ```python
  import onnx
  onnx_model = onnx.load("fashion_mnist_model.onnx")
  onnx.checker.check_model(onnx_model)
  ```

+ 创建推理Session

  ```python
  import onnxruntime as ort
  import numpy as np
  x, y = test_data[0][0], test_data[0][1]
  ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
  outputs = ort_sess.run(None, {'input': x.numpy()})
  
  # Print Result 
  predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
  print(f'Predicted: "{predicted}", Actual: "{actual}"')
  ```

# 3 Python API

不同的深度学习框架导出ONNX模型之后，都可以共用一套 ONNX Python API 推理接口，文档详见：

[Python API Reference Docs](https://onnxruntime.ai/docs/api/python/api_summary.html)

## 3.1 Overview

ONNX运行时以ONNX图形格式或ORT格式(用于内存和磁盘受限的环境)在模型上加载和运行推断。

模型使用和生成的数据可以按照最适合您的场景的方式指定和访问。

## 3.2 加载并运行模型

`InferenceSession` 是ONNX运行时的主要类。它用于加载和运行ONNX模型，以及指定环境和应用程序配置选项。

```python
session = onnxruntime.InferenceSession('model.onnx')
outputs = session.run([output names], inputs)
```

在下面的例子中，如果CUDA执行提供商ONNX运行时中有一个内核在GPU上执行。如果不是，则在CPU上执行内核。

```python
session = onnxruntime.InferenceSession(model,
              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

其他的 providers 列表详见 ：[ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/) ，如：

+ CUDA
+ OpenVINO
+ TensorRT
+ SNPE
+ 等等

自 ONNX Runtime 1.10 以后，必须显示的指定 Execution Provider（仅在运行在CPU上时，才可以不显式的设置 provider 参数）。

## 3.3 数据输入输出

ONNX Runtime 使用 `OrtValue` 类，来处理数据。

### 3.3.1 CPU

默认情况下，ONNX运行时总是将输入和输出放在CPU上。如果输入或输出是在CPU以外的设备上消耗和产生的，那么将数据放在CPU上可能不是最优的，因为它会引入CPU和设备之间的数据拷贝。

```python
import cv2
import numpy as np
import onnxruntime

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.

# X is numpy array on cpu
ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img)
print(ortvalue.device_name()) # 设备名称 ： CPU
print(ortvalue.shape()) # 数据维度 ： [1, 3, 224, 224]
print(ortvalue.data_type()) # 数据类型 : tensor(float)
print(ortvalue.is_tensor()) # True
print(np.array_equal(ortvalue.numpy(), img)) # ortvalue 和原始的 ndarray是否相等 : True

session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
results = session.run(["output"], {"input": ortvalue})
print(np.argmax(results)) # 574
```

### 3.3.2 其他设备

ONNX运行时支持自定义数据结构，支持所有的ONNX数据格式，允许用户将这些数据放在设备上，例如，在CUDA支持的设备上。在ONNX Runtime中，这叫做IOBinding。

要使用IOBinding特性，需要将 `inferencessession.run()` 替换为 `inferencessession.run_with_iobinding()` 。

+ 指定输入数据在GPU上，如果不指定输出数据的设备，则输出数据在CPU上：

```python
import cv2
import numpy as np
import onnxruntime

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.

session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

io_binding = session.io_binding()

# 输入数据在 cuda 0 上
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())

# 未指定输出数据的设备
io_binding.bind_output('output')

session.run_with_iobinding(io_binding)
print(io_binding.get_outputs()[0].device_name()) # CPU
results = io_binding.copy_outputs_to_cpu()[0]
print(np.argmax(results)) # 574
```

+ 指定输入数据在GPU上，同时也指定输出数据在 GPU上：

```python
import cv2
import numpy as np
import onnxruntime

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.

session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

io_binding = session.io_binding()

# 输入数据在 cuda 0 上
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())

# 输出数据在 cuda 0 上
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([1, 1000], np.float32, 'cuda', 0)
io_binding.bind_output(name='output', device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())

session.run_with_iobinding(io_binding)
print(io_binding.get_outputs()[0].device_name()) # cuda
results = io_binding.copy_outputs_to_cpu()[0]
print(np.argmax(results)) # 574
```

+ 上一个例子在绑定 输出数据时，**需要指定输出数据的 shape** ：`Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([1, 1000], np.float32, 'cuda', 0)` 。然而，对于动态输出的情况，也可以采用下方的示例实现动态的输出，并且输入输出都在 CUDA 上：

  ```python
  import cv2
  import numpy as np
  import onnxruntime
  
  img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
  img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
  img = img[None, ...].astype(np.float32) / 255.
  
  session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  
  io_binding = session.io_binding()
  
  # 输入数据在 cuda 0 上
  X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
  io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
  
  # 输出数据在 cuda 0 上, 但不指定输出的shape
  io_binding.bind_output(name='output', device_type='cuda')
  
  session.run_with_iobinding(io_binding)
  print(io_binding.get_outputs()[0].device_name()) # cuda
  results = io_binding.copy_outputs_to_cpu()[0]
  print(np.argmax(results)) # 574
  ```

+ 上述例子中，首先使用 `onnxruntime.OrtValue.ortvalue_from_numpy` 或者 `onnxruntime.OrtValue.ortvalue_from_shape_and_type` 定义 `OrtValue`，指定了设备和设备id。之后又使用 `io_binding.bind_input` 或 `io_binding.bind_output` 来绑定，还需要指定设备名称和设备id，略显多余。因此，还可以直接使用 `OrtValue` 最为输入输出，同时绑定设备：

  + 静态输出 shape

  ```python
  import cv2
  import numpy as np
  import onnxruntime
  
  img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
  img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
  img = img[None, ...].astype(np.float32) / 255.
  
  session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  
  
  # 输入数据在 cuda 0 上
  X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
  Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([1, 1000], np.float32, 'cuda', 0)
  
  io_binding = session.io_binding()
  io_binding.bind_ortvalue_input('input', X_ortvalue)
  io_binding.bind_ortvalue_output('output', Y_ortvalue)
  
  session.run_with_iobinding(io_binding)
  print(io_binding.get_outputs()[0].device_name()) # cuda
  results = io_binding.copy_outputs_to_cpu()[0]
  print(np.argmax(results)) # 574
  ```

  + 动态输出shape

  ```python
  import cv2
  import numpy as np
  import onnxruntime
  
  img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
  img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
  img = img[None, ...].astype(np.float32) / 255.
  
  session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  
  
  # 输入数据在 cuda 0 上
  X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
  
  io_binding = session.io_binding()
  io_binding.bind_ortvalue_input('input', X_ortvalue)
  io_binding.bind_output(name='output', device_type='cuda')
  
  session.run_with_iobinding(io_binding)
  print(io_binding.get_outputs()[0].device_name()) # cuda
  results = io_binding.copy_outputs_to_cpu()[0]
  print(np.argmax(results)) # 574
  ```

### 3.3.3 使用PyTorch Tensor作为输入输出

上述例子中，输出数据类型都是 `np.ndarray` ，ONNX 也支持输出 PyTorch Tensor：

```python
import cv2
import numpy as np
import onnxruntime
import torch

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.
img = torch.from_numpy(img)

session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

io_binding = session.io_binding()

x_tensor = img.contiguous().cuda()
y_tensor = torch.empty([1, 1000], dtype=torch.float32, device='cuda:0').contiguous()

io_binding.bind_input(
    name='input',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(x_tensor.shape),
    buffer_ptr=x_tensor.data_ptr(),
    )

io_binding.bind_output(
    name='output',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(y_tensor.shape),
    buffer_ptr=y_tensor.data_ptr(),
)

session.run_with_iobinding(io_binding)
print(io_binding.get_outputs()[0].device_name()) # cuda
results = io_binding.copy_outputs_to_cpu()[0]
print(type(results)) # <class 'numpy.ndarray'>
print(np.argmax(results)) # 574
```

+ 需要注意：上述代码中 `x_tensor,y_tensor ` 需要提前放到与 `io_binding` 相同的设备上:

  ```python
  x_tensor = img.contiguous().cuda()
  y_tensor = torch.empty([1, 1000], dtype=torch.float32, device='cuda:0').contiguous()
  
  # 下方写法会报错
  x_tensor = img.contiguous()
  y_tensor = torch.empty([1, 1000], dtype=torch.float32).contiguous()
  ```

+ 虽然输入输出都指定为了 torch.tensor，但是ONNX实际的输出还是 np.ndarray，不知是否有问题？

+ `io_binding` 的 `element_typ` 都必须写成 numpy 的数据格式，如 `np.float32`。写成 `torch.float32` 会报错。

+ 该例子应该仅为了说明可以接受 torch.tensor 作为输入，而输出还是numpy。

由于上个实验发现，尽管指定了 `y_tensor` 的数据类型为 torch.tensor，但是输出还是numpy。所以实验发现，直接不指定 `y_tensor` 为 torch.tensor也可以：

```python
import cv2
import numpy as np
import onnxruntime
import torch

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.
img = torch.from_numpy(img)

session = onnxruntime.InferenceSession('resnet18.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

io_binding = session.io_binding()

x_tensor = img.contiguous().cuda()

io_binding.bind_input(
    name='input',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(x_tensor.shape),
    buffer_ptr=x_tensor.data_ptr(),
    )

io_binding.bind_output(
    name='output',
    device_type='cuda',
    device_id=0,
)

session.run_with_iobinding(io_binding)
print(io_binding.get_outputs()[0].device_name()) # cuda
results = io_binding.copy_outputs_to_cpu()[0]
print(type(results)) # <class 'numpy.ndarray'>
print(np.argmax(results)) # 574
```



# 4 Execution Provider

ONNXRuntime还通过集成了许多硬件加速器，包括CUDA, TensorRT, OpenVINO, CoreML和NNAPI等，这取决于目标硬件平台。

 