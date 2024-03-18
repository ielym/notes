# 1 Support Matrix

查阅 ：[Support Matrix ](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)

包含：

+ 支持的平台和软件(Linux上支持哪些CUDA，cuDNN版本等) [Features for Platforms and Software](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#platform-matrix) 
+ 支持的 TensorRT 层 [Layers and Features](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-matrix)
+ 不同 TensorRT 层支持的精度 [Layers and Precision](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-precision-matrix)
+ 不同硬件平台上支持的精度（Oriin,V100等）[Hardware and Precision](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)
+  [Layers for Flow-Control Constructs](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-flow-control-constructs)
+ 不同平台的计算能力 [Compute Capability Per Platform](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#compute-capability-platform)
+ 不同平台的C++，python 版本 [Software Versions Per Platform](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#software-version-platform)
+ 支持的 ONNX 的 opset 及可以使用的精度 [ONNX Operator Support](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#supported-ops)



# 2 Installation

更多关于 `TensorRT` 在不同版本、不同平台上的安装、更新、卸载等方法，参考文档 ：[Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

+ 安装 `PyCUDA` [Installing PyCUDA](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda)

  ```bash
  python3 -m pip install numpy
  python3 -m pip install 'pycuda<2021.1'
  ```

+ 安装 `PyTroch`  The PyTorch examples have been tested with [PyTorch 1.9.0](https://github.com/pytorch/pytorch/releases/tag/v1.9.0), but may work with older versions.

+ 安装 `ONNX` The TensorRT ONNX parser has been tested with [ONNX 1.9.0](https://github.com/onnx/onnx/releases/tag/v1.9.0) and supports opset 14

+ 下载 `TensorRT`

  + 前往  https://developer.nvidia.com/tensorrt
  + 点击 `GET STARED` ，然后点击 `Download Now` (需要登陆)
  + 在 `Avaliable Versions` 选择支持的 `TensorRT` 版本（参考 Support Matrix）
  + 填问卷
  + 选择 `TensorRT` 版本，以及操作系统，以及 CUDA 版本对应的包（`.zip` for windows, `beb` for ubuntu）。
  + 点击，就开始下载了。

## 2.1 Windows

+ 下载 `.zip` 包，如 `TensorRT-8.4.3.1.Windows10.x86_64.cuda-11.6.cudnn8.4.zip` 

  + `8.x.x.x` 是 `TensorRT` 版本
  + `cuda-x.x` 是 CUDA 版本。需要注意，这里的 CUDA 版本可能和实际的CUDA版本不一致。不过只需要在选择下载包的时候注意提示信息中支持的 CUDA 版本即可。
  + `cudnn8.x` 是 cuDNN 版本

+ 选择一个路径，把 `.zip` 包解压到该路径，如 `P:\Tensort`

+ 添加 `TensorRT library`  文件夹到系统变量，有两种方式：

  + 我的电脑->右键属性->高级系统设置->环境变量->系统变量->选择`Path`并编辑->新建->粘贴刚才解压的 `lib` 的路径,如 `P:\Tensort\TensorRT-8.4.3.1\lib` 。
  + 直接把  `P:\Tensort\TensorRT-8.4.3.1\lib` 文件夹内的所有文件，复制到 CUDA 的安装目录中，如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin` 

+ 在解压的 `TesorRT` 文件夹中，有一个 python  文件夹，如 `P:\Tensort\TensorRT-8.4.3.1\python` ，进入该目录，发现有一堆 `whl` ，选择版本正确的 `whl` 进行安装：

  ```bash
  pip install tensorrt-*-cp3x-none-win_amd64.whl
  ```

+ 在解压的 `TesorRT` 文件夹中，有一个 `graphsuregeon` 文件夹，如 `P:\Tensort\TensorRT-8.4.3.1\graphsurgeon` ，进入该目录，发现有一个 `whl` ，安装：

  ```bash
  pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
  ```

+ 在解压的 `TesorRT` 文件夹中，有一个 `uff` 文件夹，如 `P:\Tensort\TensorRT-8.4.3.1\uff` ，进入该目录，发现有一个 `whl` ，安装：

  ```bash
  pip install uff-0.6.9-py2.py3-none-any.whl
  ```

+ 在解压的 `TesorRT` 文件夹中，有一个 `onnx_graphsurgeon` 文件夹，如 `P:\Tensort\TensorRT-8.4.3.1\onnx_graphsurgeon` ，进入该目录，发现有一个 `whl` ，安装：

  ```bash
  pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
  ```

+ 由于添加了系统环境变量，需要重启电脑。

## 2.2 Ubuntu

+ 下载  `.deb` 包，如 `nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.3.1-ga-20220813_1-1_amd64.deb`  (不建议这种方式，建议 [Tar File Installation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar))

  + `ubuntuxx04` 是OS 版本
  + `cuda-x.x` 是 CUDA 版本。需要注意，这里的 CUDA 版本可能和实际的CUDA版本不一致。不过只需要在选择下载包的时候注意提示信息中支持的 CUDA 版本即可。

  + `trt8.x.x.x` 是 `TensorRT` 版本
  + `yyyymmdd` 是包创建的时间

+ 安装 `deb` 包：

  ```bash
  sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.3.1-ga-20220813_1-1_amd64.deb
  ```

  可能会提示：

  ```
  sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.3.1-ga-20220813/c1c4ee19.pub
  ```

  按照提示执行即可

+ apt 安装 tensorrt

  ```bash
  sudo apt-get update
  sudo apt-get install tensorrt
  ```

+ (For Python3) :

  ```bash
  sudo apt-get install python3-libnvinfer-dev
  ```

+ 如果运行的程序需要 ONNX `graphsurgeon`，则安装：

  ```bash
  pip3 install numpy onnx
  sudo apt-get install onnx-graphsurgeon
  ```

+ 验证：

  ```bash
  dpkg -l | grep TensorRT
  ```

  如果成功，将会看到如下信息：

  ```
  ii  libnvinfer-bin                                              8.4.3-1+cuda11.6                  amd64        TensorRT binaries
  ii  libnvinfer-dev                                              8.4.3-1+cuda11.6                  amd64        TensorRT development libraries and headers
  ii  libnvinfer-plugin-dev                                       8.4.3-1+cuda11.6                  amd64        TensorRT plugin libraries
  ii  libnvinfer-plugin8                                          8.4.3-1+cuda11.6                  amd64        TensorRT plugin libraries
  ii  libnvinfer-samples                                          8.4.3-1+cuda11.6                  all          TensorRT samples
  ii  libnvinfer8                                                 8.4.3-1+cuda11.6                  amd64        TensorRT runtime libraries
  ii  libnvonnxparsers-dev                                        8.4.3-1+cuda11.6                  amd64        TensorRT ONNX libraries
  ii  libnvonnxparsers8                                           8.4.3-1+cuda11.6                  amd64        TensorRT ONNX libraries
  ii  libnvparsers-dev                                            8.4.3-1+cuda11.6                  amd64        TensorRT parsers libraries
  ii  libnvparsers8                                               8.4.3-1+cuda11.6                  amd64        TensorRT parsers libraries
  ii  onnx-graphsurgeon                                           8.4.3-1+cuda11.6                  amd64        ONNX GraphSurgeon for TensorRT package
  ii  python3-libnvinfer                                          8.4.3-1+cuda11.6                  amd64        Python 3 bindings for TensorRT
  ii  python3-libnvinfer-dev                                      8.4.3-1+cuda11.6                  amd64        Python 3 development package for TensorRT
  ii  tensorrt                                                    8.4.3.1-1+cuda11.6                amd64        Meta package for TensorRT
  ```

+ 之后，还需要 pip 安装：

  ```bash
  pip3 install --upgrade setuptools pip
  pip3 install nvidia-pyindex
  pip3 install --upgrade nvidia-tensorrt
  ```

  

  



