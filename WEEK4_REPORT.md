# 第四周实验报告：多策略对比、消融式分析与可视化

## 一、实验目的

第四周不再继续增加新的泛化方法，而是对前三周已经完成的实验进行统一整理、横向比较和机制分析。重点回答以下问题：

1. 不同泛化策略是否能够稳定改善跨数据集性能；
2. 哪些方法只对特定目标域有效，哪些方法具有更稳定的三域泛化能力；
3. 同域性能与跨域性能之间是否存在明显差异；
4. 当前项目中最适合作为最终推荐方案的方法是什么；
5. 各方法的性能变化能否从其作用机制上得到合理解释。

所有方法均以 OULU-NPU 为源数据集，并在 CASIA、MSU-MFSD、Replay-Attack 三个未见目标域上进行 Source-only 测试。目标域不参与训练、微调或参数选择。

---

## 二、对比方法

| 方法 | 主要改动 | 作用层面 |
|---|---|---|
| Baseline | ResNet18 + 基础增强 | 基础模型 |
| Strong | 随机裁切、颜色扰动、灰度化、模糊、随机擦除 | 像素空间 |
| Appearance | 颜色、亮度、对比度、灰度、清晰度等外观随机化 | 像素/成像风格 |
| MixStyle | 在浅层特征中混合样本均值和方差 | 特征统计空间 |
| Fourier | 同类别样本之间进行低频幅度谱混合并保留相位 | 频域 |

这五组实验可视为单变量策略对照：模型骨干、源数据、训练划分和跨域评价协议保持一致，主要改变泛化增强机制。

---

## 三、同域性能

| 方法 | OULU-NPU Accuracy | OULU-NPU AUC | OULU-NPU EER |
|---|---:|---:|---:|
| Baseline | 99.8161% | 99.9999% | 0.0526% |
| Strong | 95.3052% | 99.9971% | 0.2104% |
| Appearance | 97.4074% | 99.9624% | 1.0721% |
| MixStyle | 98.5898% | 99.9914% | 0.5920% |
| Fourier | 98.5373% | 99.9715% | 0.8814% |

所有方法在 OULU-NPU 同域测试中仍保持较高 AUC，但 Accuracy 和 EER 会因增强方式不同而产生一定变化。这说明源域性能很容易保持在较高水平，但这并不能代表模型具备跨数据集泛化能力。

---

## 四、跨数据集完整结果

### 4.1 CASIA

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 28.0648% | 23.4007% | 69.8964% |
| Strong | **61.2061%** | **41.9301%** | **55.6456%** |
| Appearance | 40.0720% | 38.3534% | 58.6650% |
| MixStyle | 38.5779% | 32.0224% | 63.1936% |
| Fourier | 33.6994% | 31.6763% | 64.5885% |

CASIA 上 Strong Augmentation 的三个指标均为当前最优。Fourier 虽然没有取得局部最优，但相对 Baseline 仍实现 Accuracy、AUC 上升和 EER 下降。

### 4.2 MSU-MFSD

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 60.9356% | 64.0805% | 41.6813% |
| Strong | 68.2259% | 60.1555% | 41.6844% |
| Appearance | 56.2080% | 56.6613% | 45.4044% |
| MixStyle | 65.6133% | 59.3304% | 42.4321% |
| Fourier | **75.2924%** | **65.7463%** | **39.5778%** |

MSU-MFSD 上 Fourier 同时取得最高 Accuracy、最高 AUC 和最低 EER，是表现最稳定的方法。

### 4.3 Replay-Attack

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 48.2725% | 39.1893% | 54.5629% |
| Strong | 49.1234% | 38.8743% | 54.8851% |
| Appearance | 55.5157% | 46.4997% | 51.1434% |
| MixStyle | 56.5870% | 48.3348% | 51.1476% |
| Fourier | **66.2087%** | **63.8448%** | **41.0995%** |

Replay-Attack 上 Fourier 的提升最明显，其中 AUC 相比 Baseline 提升超过 24 个百分点，EER 降低超过 13 个百分点。

---

## 五、三域宏平均

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER |
|---|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% |
| Strong | **59.5185%** | 46.9866% | 50.7384% |
| Appearance | 50.5986% | 47.1715% | 51.7376% |
| MixStyle | 53.5927% | 46.5625% | 52.2578% |
| Fourier | 58.4002% | **53.7558%** | **48.4219%** |

从宏平均看：

- Strong 的平均 Accuracy 略高于 Fourier；
- Fourier 的平均 AUC 最高；
- Fourier 的平均 EER 最低；
- 因此，如果更加重视跨域排序能力和错误率，Fourier 的总体表现优于其他方法。

---

## 六、三域一致性分析

为了避免只依据某一个数据集上的局部最好结果判断方法优劣，本项目增加一个更严格的一致性标准：相对于 Baseline，在同一个目标域上同时满足 Accuracy 上升、AUC 上升、EER 下降，才认为该目标域获得完整改善。

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 完整改善数量 |
|---|---|---|---|---:|
| Strong | 是 | 否 | 否 | 1/3 |
| Appearance | 是 | 否 | 是 | 2/3 |
| MixStyle | 是 | 否 | 是 | 2/3 |
| Fourier | **是** | **是** | **是** | **3/3** |

Fourier 是目前唯一达到三个目标域完整一致改善的方法。

---

## 七、Fourier 相对 Baseline 的提升幅度

| 目标域 | Accuracy 提升 | AUC 提升 | EER 降低 |
|---|---:|---:|---:|
| CASIA | +5.6346 pp | +8.2756 pp | 5.3079 pp |
| MSU-MFSD | +14.3568 pp | +1.6658 pp | 2.1035 pp |
| Replay-Attack | +17.9362 pp | +24.6555 pp | 13.4634 pp |

三个目标域宏平均相比 Baseline：

- Accuracy：+12.6425 个百分点；
- AUC：+11.5323 个百分点；
- EER：降低 6.9583 个百分点。

这说明 Fourier 的优势不依赖某一个特定目标域，而是具有更稳定的跨域改善趋势。

---

## 八、消融式机制分析

### 8.1 Strong Augmentation

Strong Augmentation 在 CASIA 上取得了最好的单域结果，但其 MSU-MFSD 和 Replay-Attack 的 AUC/EER 没有同步改善。这表明增加增强强度并不等于增加域泛化能力。对于 FAS，过强随机裁切、模糊和随机遮挡可能破坏与攻击介质相关的细粒度纹理。

### 8.2 Appearance Augmentation

Appearance 主要改变颜色、亮度、灰度、对比度和清晰度。它能够显著改善 Replay-Attack，同时也改善 CASIA，但在 MSU-MFSD 上三个指标均下降。这说明部分跨域差异确实来自成像外观，但不同目标域的分布偏移并不完全由颜色和外观统计决定。

### 8.3 MixStyle

MixStyle 通过打乱浅层特征统计缓解域风格依赖。在 CASIA 和 Replay-Attack 上取得改善，但在 MSU-MFSD 上 AUC 和 EER 低于 Baseline。因此，浅层均值和方差能够表示一部分域风格，但仍不足以覆盖不同数据集之间的全部攻击和成像差异。

### 8.4 Fourier

Fourier 方法保留原样本相位，并只对同类别样本的低频幅度谱进行混合。低频幅度与图像整体亮度、颜色、成像风格和设备统计具有较强关系，而相位更多保留空间结构信息。相比单纯在像素空间进行强扰动，这种方式更有针对性地削弱域相关外观统计，同时尽量保留人脸结构和与真假判别有关的局部信息。

当前实验结果与这一设计动机相符：Fourier 是唯一在三个未见目标域上均同时改善 Accuracy、AUC 和 EER 的方法。

需要强调的是，本项目实现的是轻量 Fourier amplitude augmentation，而不是完整复现某一篇频域 FAS 方法的全部网络与损失函数。

---

## 九、可视化内容

运行第四周分析脚本后，将自动生成：

```text
outputs_week4/
├── week4_all_results.csv
├── week4_macro_summary.csv
├── week4_domain_best.csv
├── week4_analysis.md
├── accuracy_comparison.png
├── auc_comparison.png
├── eer_comparison.png
├── macro_accuracy.png
├── macro_auc.png
├── macro_eer.png
└── improvement_heatmap.png
```

最终答辩中建议优先使用：

1. `auc_comparison.png`：展示五种方法在三个目标域上的 AUC；
2. `eer_comparison.png`：展示 EER 的降低情况；
3. `improvement_heatmap.png`：直观显示不同方法相对 Baseline 的改善和退化；
4. `macro_auc.png` / `macro_eer.png`：说明 Fourier 的整体泛化优势。

---

## 十、第四周结论

第四周通过统一对比 Baseline、Strong Augmentation、Appearance Augmentation、MixStyle 和 Fourier 五组方法，进一步验证了 FAS 跨数据集泛化问题具有明显的目标域差异。像素级增强、外观随机化和浅层特征统计混合均能够改善部分目标域，但效果缺乏一致性。

Fourier 低频幅度扰动是当前唯一在 CASIA、MSU-MFSD 和 Replay-Attack 三个未见目标域上同时实现 Accuracy 上升、AUC 上升和 EER 下降的方法，并取得最高的三域平均 AUC 和最低的三域平均 EER。因此，本项目将 Fourier 作为当前推荐的泛化增强方案。

但 Fourier 并未彻底解决跨域问题，尤其在 CASIA 上 AUC 仍低于 0.5。最终实验结论应表述为“频域幅度增强能够稳定改善跨数据集泛化”，而不能表述为“已经解决跨数据集泛化问题”。

完成本周后，项目已经具备进入第五周最终整理和答辩阶段的实验基础：Baseline、跨域问题验证、多种泛化策略对比、消融式机制分析、完整评价指标以及可视化结果均已具备。
