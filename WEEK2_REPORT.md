# 第二周实验报告：跨数据集泛化测试

## 一、第二周目标

第二周目标是验证第一周 OULU-NPU Baseline 在未参与训练的数据集上的直接泛化能力。核心问题是：模型在源域中接近满分的同域性能，能否迁移到不同设备、环境和攻击介质的数据分布。

实验采用 Source-only 设置：

- 源域：OULU-NPU；
- 目标域：CASIA、MSU-MFSD、Replay-Attack；
- 目标域不参与训练、微调或参数更新；
- 使用第一周训练得到的 ResNet18 Baseline 直接测试。

---

## 二、第一周同域参考

| 训练数据集 | 测试数据集 | Accuracy | AUC | EER |
|---|---|---:|---:|---:|
| OULU-NPU | OULU-NPU | 99.8161% | 99.9999% | 0.0526% |

该结果作为第二周跨域性能下降的参考基准。

---

## 三、跨数据集正式结果

| 训练数据集 | 测试数据集 | 测试图像数 | Accuracy | AUC | EER | EER Threshold |
|---|---|---:|---:|---:|---:|---:|
| OULU-NPU | CASIA | 11,110 | 28.0648% | 23.4007% | 69.8964% | 0.666826 |
| OULU-NPU | MSU-MFSD | 4,019 | 60.9356% | 64.0805% | 41.6813% | 0.251069 |
| OULU-NPU | Replay-Attack | 19,508 | 48.2725% | 39.1893% | 54.5629% | 0.006060 |

目标数据集图像数量与正式数据审计一致：CASIA 11,110 张、MSU-MFSD 4,019 张、Replay-Attack 19,508 张。

---

## 四、相对同域 Baseline 的性能变化

| 测试数据集 | Accuracy 下降 | AUC 下降 | EER 上升 |
|---|---:|---:|---:|
| CASIA | 71.7513 pp | 76.5992 pp | 69.8438 pp |
| MSU-MFSD | 38.8805 pp | 35.9194 pp | 41.6287 pp |
| Replay-Attack | 51.5436 pp | 60.8106 pp | 54.5103 pp |

三个目标域均出现显著性能下降，说明第一周接近满分的源域表现无法直接迁移到未见数据分布。

---

## 五、结果分析

### 5.1 CASIA

CASIA 是三个目标域中退化最明显的一个：Accuracy 仅 28.0648%，AUC 为 23.4007%，EER 达到 69.8964%。AUC 显著低于 0.5，说明在当前 Live-positive 评分约定下，模型对部分样本的排序关系出现严重偏移甚至近似反转。

### 5.2 MSU-MFSD

MSU-MFSD 的结果相对最好，Accuracy 60.9356%、AUC 64.0805%、EER 41.6813%，说明其数据分布与 OULU-NPU 的差异相对较小，但泛化性能仍远低于同域结果。

### 5.3 Replay-Attack

Replay-Attack 的 Accuracy 为 48.2725%、AUC 为 39.1893%、EER 为 54.5629%。AUC 低于 0.5，表明模型仍存在明显的跨域排序失效。

---

## 六、为什么跨域性能下降

可能的域偏移来源包括：

- 摄像设备不同；
- 光照与背景环境不同；
- 图像质量、压缩和重采样不同；
- 屏幕或打印介质不同；
- 颜色、对比度和清晰度统计不同；
- 不同数据集的攻击类型与采集协议不同；
- 人脸区域预处理方式可能存在差异。

因此，ResNet18 在单一 OULU-NPU 源域训练时可能同时学习了真正的 Live/Spoof 判别线索和数据集特有的颜色、纹理、设备或压缩特征。后者在新数据集上并不稳定。

---

## 七、为什么跨域分析不能只看 Accuracy

三个目标域的 EER Threshold 差异非常大：

```text
CASIA         0.666826
MSU-MFSD      0.251069
Replay-Attack 0.006060
```

这说明模型预测概率在不同目标域存在明显的分布和校准偏移。固定 `0.5` 阈值下的 Accuracy 会受到这种偏移影响，因此跨域分析需要联合观察：

- Accuracy；
- AUC；
- EER；
- ROC Curve；
- Confusion Matrix。

其中 AUC 和 EER 对判断跨域区分能力尤其重要。

---

## 八、运行与正式产物

运行：

```bash
bash scripts/run_week2_cross_dataset.sh
```

正式跨域结果位于：

```text
outputs_cross/
├── OULU-NPU_to_CASIA/
├── OULU-NPU_to_MSU-MFSD/
├── OULU-NPU_to_Replay-Attack/
├── cross_dataset_summary.csv
└── cross_dataset_summary.md
```

各目标域目录包含跨域指标、ROC 和 Confusion Matrix 等结果。

---

## 九、第二周结论

第二周实验明确证明：**同域高性能不等于跨数据集高泛化。** ResNet18 在 OULU-NPU 同域测试中达到 99.8161% Accuracy 和 99.9999% AUC，但直接迁移到 CASIA、MSU-MFSD 和 Replay-Attack 后性能均显著下降。

这说明模型存在明显的源域依赖，后续工作的重点不应继续追求 OULU-NPU 同域精度，而应通过数据增强、特征统计扰动或频域方法降低模型对源域外观统计的过度依赖。因此第三周进入跨数据集泛化增强实验。