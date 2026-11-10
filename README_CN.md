# CESAD

训练、推理和评估分别使用 `train.py`、`inference.py` 和 `evaluate.py`。
核心实现位于 `cesad/`，两种源域的配置模板位于 `configs/`，原始实验代码保留在
`research/`。运行前修改 YAML 中的本地路径，具体命令见 [README](README.md)。

本上传版本不包含独立的复现审计、权重或技术来源说明目录。数据、模型文件及大型
缓存不随代码打包；必要的第三方许可证保留。整理没有改动原始实验结果或论文。
