[图像分析](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Analyzer#analyze(androidx.camera.core.ImageProxy))用例为您的应用提供可供 CPU 访问的图像，您可以对这些图像执行图像处理、计算机视觉或机器学习推断。应用会实现对每个帧运行的 [`analyze()`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Analyzer#analyze(androidx.camera.core.ImageProxy)) 方法。

# 1 操作模式

很多情况下，分析的过程速度较慢，无法满足 CameraX 的帧速率要求。这种情况下只能选择丢帧，有两种方式进行丢帧。

简单来说：

+ 非阻塞：如果分析好了，就用分析好的帧。如果没分析好，就用最新的图像。
+ 阻塞：每帧都处理完成之后，再显示。

## 1.1 非阻塞

非阻塞是默认的丢帧方式。

在该模式下，执行器始终会将最新的图像缓存到图像缓冲区（与深度为 1 的队列相似），与此同时，应用会分析上一个图像。如果 CameraX 在应用完成处理之前收到新图像，则新图像会保存到同一缓冲区，并覆盖上一个图像。 请注意，在这种情况下，`ImageAnalysis.Builder.setImageQueueDepth()` 不起任何作用，缓冲区内容始终会被覆盖。您可以通过使用 [`STRATEGY_KEEP_ONLY_LATEST`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis#STRATEGY_KEEP_ONLY_LATEST) 调用 `setBackpressureStrategy()` 来启用该非阻塞模式。如需详细了解执行器的相关影响，请参阅 [`STRATEGY_KEEP_ONLY_LATEST`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis#STRATEGY_KEEP_ONLY_LATEST) 的参考文档

## 1.2 阻塞

在该模式下，内部执行器可以向内部图像队列添加多个图像，并仅在队列已满时才开始丢帧。系统会在整个相机设备上进行屏蔽：如果相机设备具有多个绑定用例，那么在 CameraX 处理这些图像时，系统会屏蔽所有这些用例。例如，如果预览和图像分析都已绑定到某个相机设备，那么在 CameraX 处理图像时，系统也会屏蔽相应预览。您可以通过将 [`STRATEGY_BLOCK_PRODUCER`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis#strategy_block_producer) 传递到 [`setBackpressureStrategy()`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Builder#setBackpressureStrategy(int)) 来启用阻塞模式。此外，您还可以通过使用 [ImageAnalysis.Builder.setImageQueueDepth()](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Builder#setImageQueueDepth(int)) 来配置图像队列深度。



如果分析器延迟低且性能高，在这种情况下用于分析图像的总时间低于 CameraX 帧的时长（例如，60fps 用时 16 毫秒），那么上述两种操作模式均可提供顺畅的总体体验。在某些情况下，阻塞模式仍非常有用，例如在处理非常短暂的系统抖动时。

如果分析器延迟高且性能高，则需要结合使用阻塞模式和较长的队列来抵补延迟。但请注意，在这种情况下，应用仍可以处理所有帧。

如果分析器延迟高且耗时长（分析器无法处理所有帧），非阻塞模式可能更为适用，因为在这种情况下，系统必须针对分析路径进行丢帧，但要让其他同时绑定的用例仍能看到所有帧。

# 2 实现

如需在您的应用中使用图像分析，请按以下步骤操作：

- 构建 [`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis) 用例。
- 创建 [`ImageAnalysis.Analyzer`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Analyzer)。
- [将分析器设为](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis#setAnalyzer(java.util.concurrent.Executor, androidx.camera.core.ImageAnalysis.Analyzer)) `ImageAnalysis`。
- 将生命周期所有者、相机选择器和 `ImageAnalysis` 用例[绑定](https://developer.android.com/reference/androidx/camera/lifecycle/ProcessCameraProvider#bindToLifecycle(androidx.lifecycle.LifecycleOwner, androidx.camera.core.CameraSelector, androidx.camera.core.UseCase...))到生命周期。

绑定后，CameraX 会立即将图像发送到已注册的分析器。 完成分析后，调用 [`ImageAnalysis.clearAnalyzer()`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis#clearAnalyzer()) 或解除绑定 `ImageAnalysis` 用例以停止分析。