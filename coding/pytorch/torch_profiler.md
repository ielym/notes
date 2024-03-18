# TORCH.PROFILER

文档连接 ：[TORCH.PROFILER](https://pytorch.org/docs/stable/profiler.html#torch-profiler)

Pytorch Profiler 可以在训练和推理的过程中获取性能度量。Profiler的 context manager API 能够更好的帮助理解模型的操作中哪些是最昂贵的。



# torch.profiler._KinetoProfile

```python
torch.profiler._KinetoProfile(*, activities=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=None)
```

+ `activities ` (*iterable*) : 传入一组设备。

  + 实参示例：
    + `torch.profiler.ProfilerActivity.CPU`
    + `torch.profiler.ProfilerActivity.CUDA`

  + 默认值：
    + `torch.profiler.ProfilerActivity.CPU`
    + (when avaliable) `torch.profiler.ProfilerActivity.CUDA`

+ `record_shapes ` (bool) : 是否保存