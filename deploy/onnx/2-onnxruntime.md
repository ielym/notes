# 简介

官方文档 ：[ONNX Runtime](https://onnxruntime.ai/docs/)

ONNX Runtime 是一个跨平台的机器学习模型加速器，具有灵活的接口来集成特定于硬件的库。

ONNX Runtime 可以和 PyTorch, Tensorflow/Keras，TFLite, scikit-learn以及其他框架一起使用。



ONNX Runtime 用于推理时，具有以下特性：

+ 改进各种ML模型的推理性能
+ 在不同的硬件和操作系统上运行
+ 训练Python，但部署到c# / c++ /Java应用程序
+ 训练并使用不同框架中创建的模型进行推理



ONNX Runtime 的工作流程如下：

+ 训练一个模型，如PyTorch等。
+ 使用ONNX运行时加载并运行模型 [使用不同语言运行ONNX Runtime](https://onnxruntime.ai/docs/tutorials/api-basics)
+ (Optional) 使用各种运行时配置或硬件加速器调优性能 [ONNX性能调优](https://onnxruntime.ai/docs/performance/tune-performance.html)

即使不进行 ONNX 性能调优，ONNX Runtime 相较于原始的ML框架，通常也具有性能改善。



**NOTE:** ONNX Runtime 也可以用于训练，如用于 Pytorch 并使用 NVIDIA GPU [文档](https://onnxruntime.ai/docs/#onnx-runtime-for-training) 。



# 1 安装 ONNX Runtime

文档 ：[Install ONNX Runtime (ORT)](https://onnxruntime.ai/docs/install/)

包括各个语言、平台的推理、训练时环境安装：

- [Python Installs](https://onnxruntime.ai/docs/install/#python-installs)
- [C#/C/C++/WinML Installs](https://onnxruntime.ai/docs/install/#cccwinml-installs)
- [Install on web and mobile](https://onnxruntime.ai/docs/install/#install-on-web-and-mobile)
- [ORT Training package](https://onnxruntime.ai/docs/install/#ort-training-package)
- [Inference install table for all languages](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages)
- [Training install table for all languages](https://onnxruntime.ai/docs/install/#training-install-table-for-all-languages)

此处不详细介绍细节，但会在其他对应的笔记中单独详细记录。