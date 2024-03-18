# 1 NVIDIA TensorRT

NVIDIA TensorRT是一个c++库，用于在NVIDIA GPU上进行高性能推断。TensorRT 专注于在GPU上推理一个已经训练好的网络。

提供了 C++ 和 Python 的 API，以帮助通过网络定义API表达深度学习模型，或通过解析器加载预定义的模型，允许TensorRT优化并在NVIDIA GPU上运行它们。

`TensorRT` 提供了 graph optimizations, layer fusion 以及其他优化，同时还利用各种高度优化的内核找到了该模型的最快实现。

# 2 安装

TensorRT 在不同平台上有多种安装方式，详见官方文档 [Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

这里仅以 `pip` 安装为例子，进行简单介绍。

+ `pip` 安装 `nvidia-tensorrt` 仅支持 `Python 3.6 - 3.10` ，以及`CUDA 11.x` 。其他版本的 `Python` 和 `CUDA` 版本不能工作。
+ 仅支持 `Linux` 操作系统和 `x86_64 CPU` 架构。
+ 最好在 `CentOS 7` 或 `Ubuntu 18.04` 及更高的版本。

## 2.1 安装步骤

+ 安装 `nvidia-pyindex` ，为了获取 `NGC™ PyPI` 仓库

  ```bash
  python3 -m pip install --upgrade setuptools pip
  python3 -m pip install nvidia-pyindex
  ```

+ 安装 TensorRT Python wheel

  ```bash
  python3 -m pip install --upgrade nvidia-tensorrt
  ```

+ 验证

  ```bash
  python3
  >>> import tensorrt
  >>> print(tensorrt.__version__)
  >>> assert tensorrt.Builder(tensorrt.Logger())
  ```

  如果不报错，就证明基本上安装对了。也可以在 [Samples](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html) 进一步验证是否成功安装。



# 3 工作流程

TensorRT 可以处理各种转换和部署工作流，哪个工作流最适合您，这取决于您特定的用例和问题设置。但所有的工作流都涉及到将模型转换为优化的表示，TensorRT将其称为引擎。为你的模型构建一个TensorRT工作流需要选择正确的部署选项，以及正确的引擎创建参数组合。

## 3.1 Basic TensorRT Workflow

使用TensorRT，必须遵循下方5个基本步骤来转换和部署模型：

1. Export the model
2. Select a batch-size
3. Select a precision
4. Convert the model
5. Deploy the model

为了对上述5个基本步骤有一个初步的理解，这里使用一个 ONNX 模型来完成 TensorRT的转换和部署。

![Deployment Process Using ONNX](imgs/deploy-process-onnx.png)

### 3.1.1 Export the Model

TensorRT 主要使用两种方式来转换模型：

+ TF-TRT uses TensorFlow SavedModels
+ The ONNX path requires that models are saved in ONNX

这里使用 ONNX。在该过程结束之后，会得到一个 `xxx.onnx` 模型（导出 onnx 文件可以参考 onnx的笔记，fp32 即可）。

### 3.1.2 Select a Batch Size

在推理时Batch-size也是对效率的一个重要影响因素。当需要更小的延时时，可以设置比较小的bs；当需要更大的吞吐量时，可以设置较大的bs。

当然，如果不确定推理时的bs，TensorRT 也提供了动态设置 BS的方式。

但是在这个例子中，以静态设置 bs 为例：

```python
BATCH_SIZE=64
```

### 3.1.3 Select a Precision

推理时通常需要比训练时更少的浮点精度，能够具有更快的计算能力和耕地的内存消耗，TensorRT支持的浮点精度包括：

+ TF32
+ FP32
+ FP16
+ INT8

```python
import numpy as np
PRECISION = np.float32
```

为了方便理解，不隐入过多量化的内容。这里就以 FP32为例。上面的代码先放在这，后面再使用。

### 3.1.4 Convert the Model

有多种方式能够把 ONNX 模型转换为 TensorRT engine。其中常用的方式是使用 `trtexec` (**t**ensor**rt** exec) 。

`trtexec` 是一个命令行工具，可以转换 ONNX 模型为 TensorRT engines 并且进行配置。

```bash
trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt
```

+ `resnet50/model.onnx` 是 ONNX 模型的路径
+ `resnet_engine.trt` 是导出的 TensorRT engine 的文件名。

`trtexec` 其他 config:

+ 示例

  ```python
  trtexec 
  --onnx=fcn-resnet101.onnx 
  --fp16 
  --workspace=64 
  --minShapes=input:1x3x256x256 
  --optShapes=input:1x3x1026x1282 
  --maxShapes=input:1x3x1440x2560 
  --buildOnly 
  --saveEngine=fcn-resnet101.engine
  ```

+ `--fp16` : 使用 fp16 精度
+ `--int8` : 使用 int 精度
+ `--best` : 使所有支持的精度达到最佳性能的每一层
+ `--workspace` : 控制构建器考虑的算法可用的最大持久暂存内存(以MB为单位)。对于给定的平台，应该根据可用性设置尽可能高的值;在运行时，TensorRT将只分配需要的，不超过最大。
+ `--minShapes` 和 `--maxShapes` 指定每个网络输入的维度范围，而 `--optShapes` 指定自动调优器应该用于优化的维度
+ `--buildOnly` ：跳过推断性能度量的请求
+ `--saveEngine` : 指定必须将序列化引擎保存到其中的文件
+ `--tacticSources` : 可以用来添加或删除策略从默认的策略来源(cuDNN, cuBLAS，和cuBLASLt)
+ `--minTiming` 和 `--avgTiming` : 分别设置策略选择的最小迭代次数和平均迭代次数
+ `--noBuilderCache` : 在TensorRT构建器中禁用层计时缓存。计时缓存通过缓存层分析信息，有助于减少构建阶段所花费的时间，并且应该适用于大多数情况。在有问题的情况下使用这个开关
+ `--timingCacheFile` : 可用于保存或加载序列化的全局定时缓存

### 3.1.5 Deploy the Model

在成功转换了 `TensorRT engine` `xxx.trt` 之后，就需要决定如何使用 `TensorRT` 来运行，有两种 `TensorRT runtime`:

+ a standalone runtime that has C++ and Python bindings
+ a native integration into TensorFlow

更多 `TensorRT runtime` 的信息，可以查阅  [Understanding TensorRT Runtimes](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/IntroNotebooks/5. Understanding TensorRT Runtimes.ipynb)

这里，使用一个简单的 `wrapper` : `ONNXClassifierWrapper`，即 standalone runtime：

1. 设置 `ONNXClassifierWrapper` 并设置精度：

   ```python
   from onnx_helper import ONNXClassifierWrapper
   N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task
   trt_model = ONNXClassifierWrapper("resnet_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
   ```

   其中，`PRECISION`， `BATCH_SIZE` 是我们之前定义的。

2. 生成一批数据：

   ```python
   dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3))
   ```

3. 把数据喂给 engine，并得到预测结果：

   ```python
   predictions = trt_model.predict(dummy_input_batch)
   ```

至此，就实现了一个非常简单的 ONNX -> TensorRT engine -> 部署 的流程。更多的用法可以参考  [API Reference](https://docs.nvidia.com/deeplearning/tensorrt/api/index.html)
