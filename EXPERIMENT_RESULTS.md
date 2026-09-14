# 实验结果记录

本文档用于持续记录 FAS 项目的各阶段实验结果，便于后续撰写实验报告、进行横向对比与复现实验。

## 第一周：单数据集 Baseline

第一周目标是使用 ResNet18 完成 Live / Spoof 二分类 Baseline，并在严格避免 Subject 泄漏的前提下完成同域测试。当前正式结果采用 OULU-NPU 数据集的 Subject-disjoint 划分结果。

| 实验编号 | 数据集 | 模型 | 划分方式 | 输入尺寸 | Accuracy | AUC | EER | EER Threshold | Test Spoof | Test Live | 错误样本数 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1-OULU-R18-001 | OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 224×224 | 99.8161% | 99.9999% | 0.0526% | 0.643187 | 7630 | 3787 | 21 |

### 第一周混淆矩阵

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

由结果可见，本次测试中 21 个 Spoof 样本被错误预测为 Live，Live 样本全部分类正确。

### 第一周实验设置

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

训练在第 8 个 epoch 触发 Early Stopping。Subject-disjoint 划分后，同一个 Subject 不会同时出现在 Train、Validation 和 Test 中，因此该结果可作为第一周正式 Baseline。

---

## 第二周：跨数据集泛化实验

第二周目标是验证第一周 Baseline 在未参与训练的新数据集上的泛化能力。实验采用 Source-only 设置：模型只在 OULU-NPU 上训练，CASIA、MSU-MFSD 和 Replay-Attack 仅作为目标域测试集，测试过程中不进行训练、微调或参数更新。

### 第二周正式结果

| 实验编号 | 训练数据集 | 测试数据集 | 模型 | 训练方式 | 测试图像数 | Accuracy | AUC | EER | EER Threshold |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| W2-OULU-CASIA-001 | OULU-NPU | CASIA | ResNet18 | Source-only | 11110 | 28.0648% | 23.4007% | 69.8964% | 0.666826 |
| W2-OULU-MSU-001 | OULU-NPU | MSU-MFSD | ResNet18 | Source-only | 4019 | 60.9356% | 64.0805% | 41.6813% | 0.251069 |
| W2-OULU-REPLAY-001 | OULU-NPU | Replay-Attack | ResNet18 | Source-only | 19508 | 48.2725% | 39.1893% | 54.5629% | 0.006060 |

### 同域与跨域对比

| 场景 | 训练数据集 | 测试数据集 | Accuracy | AUC | EER |
|---|---|---|---:|---:|---:|
| 同域 Baseline | OULU-NPU | OULU-NPU | 99.8161% | 99.9999% | 0.0526% |
| 跨域 | OULU-NPU | CASIA | 28.0648% | 23.4007% | 69.8964% |
| 跨域 | OULU-NPU | MSU-MFSD | 60.9356% | 64.0805% | 41.6813% |
| 跨域 | OULU-NPU | Replay-Attack | 48.2725% | 39.1893% | 54.5629% |

### 性能下降幅度

相对于 OULU-NPU 同域 Baseline：

| 测试数据集 | Accuracy 下降 | AUC 下降 | EER 上升 |
|---|---:|---:|---:|
| CASIA | 71.7513 个百分点 | 76.5992 个百分点 | 69.8438 个百分点 |
| MSU-MFSD | 38.8805 个百分点 | 35.9194 个百分点 | 41.6287 个百分点 |
| Replay-Attack | 51.5436 个百分点 | 60.8106 个百分点 | 54.5103 个百分点 |

### 第二周结果分析

第一周同域实验中，ResNet18 在 OULU-NPU 上取得了接近完全正确的分类结果，但当模型直接迁移到未参与训练的数据集后，性能出现显著下降，说明模型在源域中学习到的判别特征具有较强的数据集依赖性。

在三个目标域中，MSU-MFSD 的跨域表现相对最好，Accuracy 为 60.9356%，AUC 为 64.0805%，EER 为 41.6813%；Replay-Attack 次之，Accuracy 为 48.2725%，AUC 为 39.1893%，EER 为 54.5629%；CASIA 的跨域性能最差，Accuracy 仅为 28.0648%，AUC 为 23.4007%，EER 达到 69.8964%。这表明不同目标数据集与 OULU-NPU 之间存在程度不同的分布偏移。

造成跨数据集性能下降的潜在因素包括摄像设备差异、光照条件差异、攻击介质差异、图像质量差异、颜色分布差异、压缩与重采样过程差异，以及不同数据集的人脸裁切和预处理差异。ResNet18 在单一源域训练时可能同时学习到真正的活体线索和数据集特有的纹理、颜色、设备或成像特征，因此这些特征在新数据集上无法稳定保持。

值得注意的是，CASIA 和 Replay-Attack 的 AUC 均低于 0.5。这不仅表示模型区分能力较弱，还说明在这些目标域上，模型输出分数与真实类别之间可能出现较强的排序反转现象，即源域中被视为 Live 的特征模式在目标域中不再具有相同含义。因此，仅观察 Accuracy 不足以完整反映跨域泛化能力，AUC 和 EER 对分析域偏移非常重要。

第二周实验验证了本项目后续研究跨数据集泛化的必要性：单数据集内极高的性能并不代表模型具有真实场景中的泛化能力。第三周应在保持相同 Source-only / Cross-dataset 评价协议的前提下，引入数据增强或其他域泛化策略，并重点观察跨域 AUC 是否提高、EER 是否下降。

### 第二周结论

本实验成功建立了 OULU-NPU 到 CASIA、MSU-MFSD 和 Replay-Attack 的跨数据集评价流程。实验结果表明，ResNet18 在 OULU-NPU 同域测试中虽然能够获得 99.8161% Accuracy 和 99.9999% AUC，但在三个未见目标域上的性能均大幅下降，说明模型存在显著的域依赖问题，跨数据集泛化能力不足。该结果为第三周开展泛化增强实验提供了直接实验依据。

### 第二周运行方式

```bash
bash scripts/run_week2_cross_dataset.sh
```

输出目录：

```text
outputs_cross/
├── OULU-NPU_to_CASIA/
├── OULU-NPU_to_MSU-MFSD/
├── OULU-NPU_to_Replay-Attack/
├── cross_dataset_summary.csv
└── cross_dataset_summary.md
```

每个目标数据集目录保存对应的 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix 等结果。

---

## 第三周：泛化增强实验

第三周在保持 ResNet18、Subject-disjoint 数据划分、优化器、训练超参数以及 Source-only 跨数据集评价协议不变的前提下，只修改训练阶段的数据增强策略，从而验证更强的数据扰动能否降低模型对单一源域外观统计特征的依赖，并提升跨数据集泛化能力。

### 强数据增强策略

第三周新增 `augmentation=strong`。训练阶段在 Baseline 的基础上使用更强的随机扰动：

- `RandomResizedCrop`：随机改变裁切区域与局部尺度；
- `RandomHorizontalFlip`：随机水平翻转；
- `ColorJitter`：增强亮度、对比度、饱和度与色调变化；
- `RandomGrayscale`：随机灰度化，降低模型对固定颜色分布的依赖；
- `GaussianBlur`：模拟成像模糊与不同清晰度；
- `RandomErasing`：随机遮挡局部区域，抑制模型过度依赖单一区域纹理。

验证集、同域测试集和跨域目标数据均不使用随机增强，保证评价协议与第一、二周一致。

### 第三周待运行实验

| 实验编号 | 训练数据集 | 测试数据集 | 模型 | 增强方式 | Accuracy | AUC | EER | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| W3-OULU-OULU-STRONG-001 | OULU-NPU | OULU-NPU | ResNet18 | Strong Augmentation | 待运行 | 待运行 | 待运行 | 同域测试 |
| W3-OULU-CASIA-STRONG-001 | OULU-NPU | CASIA | ResNet18 | Strong Augmentation | 待运行 | 待运行 | 待运行 | Source-only |
| W3-OULU-MSU-STRONG-001 | OULU-NPU | MSU-MFSD | ResNet18 | Strong Augmentation | 待运行 | 待运行 | 待运行 | Source-only |
| W3-OULU-REPLAY-STRONG-001 | OULU-NPU | Replay-Attack | ResNet18 | Strong Augmentation | 待运行 | 待运行 | 待运行 | Source-only |

### 第三周评价重点

第三周不能只观察同域 Accuracy。最关键的判断依据是第二周跨域 Baseline 与第三周 Strong Augmentation 的对比：

1. 跨域 AUC 是否整体提高；
2. 跨域 EER 是否整体下降；
3. Accuracy 是否在多数目标域上提升；
4. 同域性能是否出现可接受范围内的下降；
5. 泛化提升是否具有一致性，而不是只对单个目标数据集有效。

如果增强后同域 Accuracy 略微下降，但多个目标域的 AUC 提高、EER 降低，则说明模型牺牲了部分源域拟合能力，换取了更好的域外泛化能力，这种变化通常更符合第三周实验目标。

### 第三周运行方式

```bash
bash scripts/run_week3_generalization.sh
```

训练模型默认保存于：

```text
outputs_week3/strong/OULU-NPU/
```

跨域结果默认保存于：

```text
outputs_week3_cross/strong/
```

全部实验完成后，可生成第二周 Baseline 与第三周 Strong Augmentation 的差值对比：

```bash
python compare_week3_results.py
```

输出：

```text
outputs_week3_cross/strong/week3_comparison.md
```

---

## 后续实验记录模板

| 实验编号 | 训练数据集 | 测试数据集 | 模型 | 方法/改动 | Accuracy | AUC | EER | 备注 |
|---|---|---|---|---|---:|---:|---:|---|
| 待补充 | 待补充 | 待补充 | ResNet18 | Baseline / 泛化增强 | - | - | - | - |
