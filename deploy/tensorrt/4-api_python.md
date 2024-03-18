 # Python API

`TensorRT` Python API 能够使开发者在基于 Python 的开发环境下使用。

官方文档 ： [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)

Sample Codes : [TensorRT Developer Guide](http://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)

来自于 developer guide 的 Python API介绍（比较简单，但有示例）：https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics

# 1 基本的使用-Demo

以 ONNX 为例。

+ 可以使用`tensorrt` 模块访问 Python API

  ```python
  import tensorrt as trt
  ```

## 1.1 Build Phase

+ 要创建构建器，您必须首先创建记录器。Python绑定包括一个简单的日志记录器实现，它将特定级别之前的所有消息记录到stdout。

  ```python
  logger = trt.Logger(trt.Logger.WARNING)
  ```

+ 作为替代，也可以通过派生自ILogger类来定义自己的日志器实现:

  ```python
  class MyLogger(trt.ILogger):
      def __init__(self):
         trt.ILogger.__init__(self)
  
      def log(self, severity, msg):
          pass # Your custom logging implementation here
  
  logger = MyLogger()
  ```

+ 构造 builder :

  ```python
  builder = trt.Builder(logger)
  ```

+ 创建 builder 之后，优化模型的第一步是创建一个网络定义:

  ```python
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  ```
  
  其中，`trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH` 表示显式的指定 batch size (这种方式也支持动态 bs) 。与之相对应的是隐式 batch mode，但是已经被废弃了（但是还能使用）。详见 [Explicit Versus Implicit Batch](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch)
  
+ 创建 ONNX 解析器

  ```python
  parser = trt.OnnxParser(network, logger)
  ```

+ 读取 ONNX 模型文件，并处理任何可能出现的错误：

  ```python
  success = parser.parse_from_file(model_path)
  for idx in range(parser.num_errors):
      print(parser.get_error(idx))
  
  if not success:
      pass # Error handling code here
  ```

  如：

  ```
  Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
  ```

  上述问题只是个警告，实际没影响。但如果非要解决，解决方式可以使用 `trtexec` （该过程和接下来的 build 一个 engine 是在干同一件事情）：

  ```
  P:\Tensort\TensorRT-8.4.3.1\bin\trtexec --onnx=resnet18.onnx --saveEngine=resnet18.trt
  ```

+ build 一个 engine

  + 创建一个构建配置，指定TensorRT应该如何优化模型

    ```python
    config = builder.create_builder_config()
    ```

  + 这个接口有很多属性，可以通过设置来控制TensorRT如何优化网络。一个重要的属性是最大工作空间大小 `maximum workspace size` 。Layer implementation 通常需要一个临时工作区，这个参数限制了网络中任何层可以使用的最大大小。如果提供的工作空间不足，可能TensorRT将无法找到一个层的实现:

    ```python
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB -> 2^10=1Kib, 2^20=1Mib 10^3=1kb, 10^20=1Mb
    ```

  + 在配置被指定之后，engine 可以被构建和序列化:

    ```python
    serialized_engine = builder.build_serialized_network(network, config)
    ```

    实际操作的过程中，这一步报错了：

    ```
    [09/04/2022-23:46:18] [TRT] [E] 4: [network.cpp::nvinfer1::Network::validate::2965] Error Code 4: Internal Error (Network has dynamic or shape inputs, but no optimization profile has been defined.)
    [09/04/2022-23:46:18] [TRT] [E] 2: [builder.cpp::nvinfer1::builder::Builder::buildSerializedNetwork::636] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
    ```

    原因是从 pytorch 导出 onnx 时，设置了动态维度。为了解决这个问题，需要使用 `trtexec` 来构建 engine，或者在导出onnx时使用静态维度。

  + 将引擎保存到文件中以备将来使用可能会使用（这一步和 trtexec 的作用相同，一个是python 接口，一个是命令行工具）：

    ```python
    with open(“resnet18.engine”, “wb”) as f:
        f.write(serialized_engine)
    ```

  + 一个好奇的地方是，使用命令行来构建 engine 时的官方示例的后缀名是 `.trt` ，而使用 Python 接口时的后缀名是 `.engine` 。这两者没有区别。

  + 需要注意的是，序列化引擎不能跨平台或TensorRT版本移植。引擎是特定于它们所基于的GPU模型。

## 1.2 Deserializing a Plan (Runtime)

+ 要执行推理，需要使用Runtime接口反序列化engine。与builder一样，运行时也需要Logger的实例:

  ```python
  runtime = trt.Runtime(logger)
  ```

+ 可以使用一个 序列化 的 buffer，如1.1节的 `serialized_engine`:

  ```python
  engine = runtime.deserialize_cuda_engine(serialized_engine)
  ```

+ 也可以从文件加载一个 engine (`.trt`, `.engine`):

  ```python
  with open(“resnet18.engine”, “rb”) as f:
      serialized_engine = f.read()
  ```

## 1.3 Performing Inference (Deploy)

+ engine 拥有优化的模型，但是执行推理需要额外的状态来进行中间激活。这是使用IExecutionContext接口完成的：

  ```python
  context = engine.create_execution_context()
  ```

+ 要执行推理，必须为输入和输出传递TensorRT缓冲区，TensorRT要求在GPU指针列表中指定缓冲区。可以使用提供的输入和输出张量的名称来查询引擎，以找到数组中正确的位置：

  ```python
  input_idx = engine[input_name] # 0
  output_idx = engine[output_name] # 1
  ```

  其中，`input_name` 和 `output_name` 就是 Pytorch 导出 ONNX 时设置的name。

  如果忘了怎么办？下方代码可以查看：

  ```python
  engine = runtime.deserialize_cuda_engine(serialized_engine)
  for idx in range(engine.num_bindings):
      is_input = engine.binding_is_input(idx)
      name = engine.get_binding_name(idx)
      op_type = engine.get_binding_dtype(idx)
      shape = engine.get_binding_shape(idx)
      print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)    
  ```

  ```
  input id: 0    is input:  True   binding name: inputtttt   shape: (1, 3, 224, 224) type:  DataType.FLOAT
  input id: 1    is input:  False   binding name: outputtttt   shape: (1, 1000) type:  DataType.FLOAT
  ```

+ 然后，创建一个GPU指针列表。例如，对于PyTorch CUDA张量，你可以使用data_ptr()方法访问GPU指针。

  ```python
  buffers = [None] * 2 # Assuming 1 input and 1 output
  buffers[input_idx] = input_ptr
  buffers[output_idx] = output_ptr
  ```

  其中， `input_ptr` 和 `output_ptr` 是什么呢？其实表示的是输入数据和输出数据的指针（这里不能再用 python 的思路来考虑问题了，既然有输入有输出，那就得先占个位置）。

  可以这样获取指针：

  ```python
  buffers = [None] * 2 # Assuming 1 input and 1 output
  buffers[input_idx] = torch.empty(size=(1, 3, 224, 224), dtype=torch.float, device='cuda:0').contiguous().data_ptr()
  output = torch.empty(size=(1, 1000), dtype=torch.float, device='cuda:0')
  buffers[output_idx] = output.data_ptr()
  ```

  需要特特特别注意：

  + 对于input，其指针直接写的是 `torch.empty(...).data_ptr()`（这里的empty是为了简单举例，实际操作时换成输入数据即可。）
  + 但是对于 output，其写法确实先定义 `output=torch.empty(...)`，然后再`output.data_prt()` 来传递指针。
  + 这是为什么呢？其实还是因为受python的毒害太重。输入数据直接获取指针，并且传进去无所谓，反正后面也用不到输入数据了（一般情况用不到了）。但是输出数据我们如果直接传进去个指针，后面我们也只能知道个指针，但是在Python中，我们怎么把这个指针指向的地址里的这个结果给取出来呢？？？？所以，先定义output，再传指针，不影响后续我们从output中取值。

+ 之后，就可以推理了：

  ```python
  context.execute_async_v2(buffers, stream_ptr)
  ```

  其中，`stream_ptr = torch.cuda.current_stream().cuda_stream `



## 1.3 升级版

对于1.1和1.2的序列化和反序列化，不需要用到模型的具体细节（代码中不用写）。但是在推理过程中，又是需要ONNX的名称，又是需要指针，并且在构建指针的过程中，还需要用到size, dtype, device等等参数。然而，如果 ONNX 不是我们自己导出的，而是别人给我们的，该怎么办呢？

1.3 节中也说过，可以用下方代码：

```python
engine = runtime.deserialize_cuda_engine(serialized_engine)
for idx in range(engine.num_bindings):
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    op_type = engine.get_binding_dtype(idx)
    shape = engine.get_binding_shape(idx)
    print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)    
```

但是，后面还是需要自己去找各种名称和值，再复制粘贴到定义 buffer，指针的函数中，太麻烦了，不如整理一个自动的小函数，来帮我们完成这些内容：

```python
import tensorrt as trt
import torch
import cv2
import numpy as np

class TensorRTModel():
    def __init__(self, engine=None, input_names=None, output_names=None):
        '''
        :param engine:
        :param input_names: 可以是None
        :param output_names: 可以是None
        '''
        self.engine = engine
        assert self.engine, "engine cannot be None, you can deserialize an engine using runtime.deserialize_cuda_engine()"

        if self.engine is not None:
            self.context = self.engine.create_execution_context()

        if input_names == None or output_names == None:
            input_names = []
            output_names = []
            for idx in range(engine.num_bindings):
                name = engine.get_binding_name(idx)
                op_type = engine.get_binding_dtype(idx)
                shape = engine.get_binding_shape(idx)

                is_input = engine.binding_is_input(idx)

                if is_input:
                    input_names.append(name)
                else:
                    output_names.append(name)
            print(input_names, output_names)
        self.input_names = input_names
        self.output_names = output_names

    def torch_info_from_trt(self, bind_idx):
        trt_shape = self.context.get_binding_shape(bind_idx)
        torch_shape = tuple(trt_shape)

        trt_device = self.engine.get_location(bind_idx)
        if trt_device == trt.TensorLocation.DEVICE:
            torch_device = torch.device("cuda")
        elif trt_device == trt.TensorLocation.HOST:
            torch_device = torch.device("cpu")
        else:
            raise TypeError(f"Unknow trt_device : {trt_device}")

        trt_dtype = self.engine.get_binding_dtype(bind_idx)
        if trt_dtype == trt.int8:
            torch_dtype = torch.int8
        elif trt.__version__ >= '7.0' and trt_dtype == trt.bool:
            torch_dtype = torch.bool
        elif trt_dtype == trt.int32:
            torch_dtype = torch.int32
        elif trt_dtype == trt.float16:
            torch_dtype = torch.float16
        elif trt_dtype == trt.float32:
            torch_dtype = torch.float32
        else:
            raise TypeError(f"Unknow trt_dtype : {trt_dtype}")

        return torch_shape, torch_device, torch_dtype

    def __call__(self, *inputs):

        buffers = [None] * (len(self.input_names) + len(self.output_names))

        for input_idx, input_name in enumerate(self.input_names):
            bind_idx = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(bind_idx, tuple(inputs[input_idx].shape))
            buffers[bind_idx] = inputs[input_idx].contiguous().data_ptr()


        # output buffers
        outputs = [None] * len(self.output_names)
        for output_idx, output_name in enumerate(self.output_names):
            bind_idx = self.engine.get_binding_index(output_name)
            torch_shape, torch_device, torch_dtype = self.torch_info_from_trt(bind_idx)
            output = torch.empty(size=torch_shape, dtype=torch_dtype, device=torch_device)

            outputs[output_idx] = output
            buffers[bind_idx] = output.data_ptr()

        stream = torch.cuda.current_stream()
        self.context.execute_async_v2(buffers, stream.cuda_stream)
        stream.synchronize()
        
        return outputs

def build_runtime(serialized_engine=None, serialized_engine_path=''):
    '''
    serialized_engine 和 serialized_engine_path 只需要传一个即可。如果两个都传，用 serialized_engine
    :param serialized_engine: 是一个buffer, 如：serialized_engine = builder.build_serialized_network(network, config)
    :param serialized_engine_path: 是 .trt 或 .engine 的文件
    :return:
    '''
    logger = trt.Logger(trt.Logger.WARNING)

    runtime = trt.Runtime(logger)

    if serialized_engine:
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    elif serialized_engine_path:
        with open(serialized_engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        raise Exception("You must define one of serialized_engine or serialized_engine_path")

    return engine

img = cv2.imread(r'S:\datasets\imagenet\train\n03445777\ILSVRC2012_val_00023215.JPEG')
img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])
img = img[None, ...].astype(np.float32) / 255.
img = torch.from_numpy(img).cuda()

engine = build_runtime(serialized_engine_path="resnet18.engine")
model = TensorRTModel(engine=engine)
outputs = model(img)
output = outputs[0]
print(torch.argmax(output[0]))
```

