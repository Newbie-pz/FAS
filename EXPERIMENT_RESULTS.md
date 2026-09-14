# 实验结果记录

本文档用于持续记录 FAS 项目的各阶段实验结果，便于后续撰写实验报告、进行横向对比与复现实验。

## 第一周：单数据集 Baseline

第一周目标是使用 ResNet18 完成 Live / Spoof 二分类 Baseline，并在严格避免 Subject 泄漏的前提下完成同域测试。当前正式结果采用 OULU-NPU 数据集的 Subject-disjoint 划分结果。

| 实验编号 | 数据集 | 模型 | 划分方式 | 输入尺寸 | Accuracy | AUC | EER | EER Threshold | Test Spoof | Test Live | 错误样本数 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1-OULU-R18-001 | OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 224×224 | 99.8161% | 99.9999% | 0.0526% | 0.643187 | 7630 | 3787 | 21 |

### 混淆矩阵

本次测试集混淆矩阵为：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

由结果可见，本次测试中 21 个 Spoof 样本被错误预测为 Live，Live 样本全部分类正确。

### 实验设置

| 参数 | 数值 |
|---|---|
| `model` | `ResNet18` |
| `image_size` | `224` |
| `batch_size` | `64` |
| `epochs` | `20` |
| `lr` | `1e-4` |
| `weight_decay` | `1e-4` |
| `patience` | `5` |
| `seed` | `42` |
| `optimizer` | `AdamW` |
| `pretrained` | `ImageNet` |
| `train_ratio` | `0.70` |
| `val_ratio` | `0.15` |
| `test_ratio` | `0.15` |

### 训练过程摘要

训练在第 8 个 epoch 触发 Early Stopping。验证集 Accuracy 在不同 epoch 间存在正常波动，其中第 3 个 epoch 的验证表现较好，说明 Subject-disjoint 划分后模型不再像早期 Video-level 划分那样出现从第一个 epoch 开始即完全正确的现象。

### 结果解释

本实验属于同数据集、同域条件下的测试，即训练集、验证集和测试集均来自 OULU-NPU，但三者 Subject 完全不重叠。Accuracy 反映固定阈值 0.5 下的分类正确率；AUC 衡量不同阈值下模型对 Live 与 Spoof 的整体排序能力；EER 表示错误接受率与错误拒绝率相等时的错误率，因此三项指标应结合分析。

虽然本次同域结果较高，但这并不能证明模型已经获得较强的跨域泛化能力。不同数据集在摄像设备、光照环境、攻击介质、图像质量和预处理过程等方面均可能存在明显分布差异，因此后续应使用跨数据集实验进一步验证模型是否学习到了具有泛化性的活体线索。

## 后续实验记录模板

| 实验编号 | 训练数据集 | 测试数据集 | 模型 | 方法/改动 | Accuracy | AUC | EER | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| 待补充 | 待补充 | 待补充 | ResNet18 | Baseline / 泛化增强 | - | - | - | - |
