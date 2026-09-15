# 第四周实验报告：多策略对比、消融式分析与可视化

> **方法命名说明**：第四周中的 `Strong Data Augmentation` 与 `Appearance Randomization` 是本项目自定义训练策略；`MixStyle` 是已有正式方法名，本项目做 ResNet18 轻量集成；`Fourier Amplitude Augmentation` 是本项目实现的轻量频域增强，不等同于完整复现某篇频域 FAS 方法。完整实现细节见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

> **数据协议说明**：第四周所有统计都基于同一套 `ProcessedData`、同一 Subject-disjoint 划分和同一 Source-only 跨域评价协议。原始视频抽帧、Face Detection 和 Face Crop 的历史参数未保留在当前仓库，数据入口与质量审计见 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)。因此第四周对比只解释当前可复现的训练、增强与评估差异。

## 一、实验目的

第四周不再继续增加新的泛化方法，而是对前三周已经完成的实验进行统一整理、横向比较和机制分析。重点回答：不同泛化策略是否能稳定改善跨数据集性能、哪些策略只在特定目标域有效、以及当前项目中哪种方案具有更好的跨域一致性。

所有方法均以 OULU-NPU 为源数据集，并在 CASIA、MSU-MFSD、Replay-Attack 三个未见目标域上进行 Source-only 测试。目标域不参与训练、微调或参数选择。

---

## 二、对比方法与名称性质

| 方法 | 名称性质 | 主要改动 | 作用层面 |
|---|---|---|---|
| Baseline | 项目基础配置 | ResNet18 + 基础增强 | 基础模型 |
| Strong Data Augmentation | **项目自定义** | 随机裁切、较强颜色扰动、灰度化、模糊、随机擦除 | 像素空间 |
| Appearance Randomization | **项目自定义** | 颜色、对比度、灰度、自动对比度、模糊、锐度随机化；不做随机裁切和擦除 | 像素/成像风格 |
| MixStyle | 已有正式方法名；本项目轻量集成 | 在 `layer1/layer2` 后混合样本均值和方差 | 特征统计空间 |
| Fourier Amplitude Augmentation | **项目轻量实现** | 同类别样本低频幅度谱混合并保留源相位 | 频域 |

### Strong Data Augmentation 实现摘要

代码参数：`--augmentation strong`。

主要设置：`RandomResizedCrop(scale=(0.80,1.00), ratio=(0.90,1.10))`、ColorJitter 强度 `0.40/0.40/0.40/0.10` 且 `p=0.80`、`RandomGrayscale(p=0.15)`、`GaussianBlur(kernel_size=5, sigma=(0.1,2.0), p=0.30)`、`RandomErasing(p=0.25)`。

### Appearance Randomization 实现摘要

代码参数：`--augmentation appearance`。

主要设置：ColorJitter 强度 `0.35/0.35/0.30/0.05` 且 `p=0.80`、`RandomGrayscale(p=0.10)`、`RandomAutocontrast(p=0.15)`、`GaussianBlur(kernel_size=3, sigma=(0.1,1.0), p=0.15)`、`RandomAdjustSharpness(sharpness_factor=0.6, p=0.10)`。该策略不使用 `RandomResizedCrop` 和 `RandomErasing`。

### MixStyle 实现摘要

本项目将 MixStyle 插在 ResNet18 `layer1` 和 `layer2` 后，默认 `p=0.5`、`alpha=0.1`，混合系数服从 `Beta(0.1,0.1)`；仅训练阶段启用。

### Fourier 实现摘要

代码参数：`--method fourier`。训练阶段以 `p=0.5` 的概率执行类内低频幅度混合，最大混合权重为 `0.35`，低频区域比例为 `0.10`。donor 只在同类别内选择：`Live↔Live`、`Spoof↔Spoof`。验证和测试阶段关闭。

---

## 三、同域性能

| 方法 | OULU-NPU Accuracy | OULU-NPU AUC | OULU-NPU EER |
|---|---:|---:|---:|
| Baseline | 99.8161% | 99.9999% | 0.0526% |
| Strong Data Augmentation | 95.3052% | 99.9971% | 0.2104% |
| Appearance Randomization | 97.4074% | 99.9624% | 1.0721% |
| MixStyle | 98.5898% | 99.9914% | 0.5920% |
| Fourier Amplitude Augmentation | 98.5373% | 99.9715% | 0.8814% |

所有方法在 OULU-NPU 同域测试中仍保持较高 AUC，但这不能代表模型具备跨数据集泛化能力。

---

## 四、跨数据集完整结果

### 4.1 CASIA

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 28.0648% | 23.4007% | 69.8964% |
| Strong Data Augmentation | **61.2061%** | **41.9301%** | **55.6456%** |
| Appearance Randomization | 40.0720% | 38.3534% | 58.6650% |
| MixStyle | 38.5779% | 32.0224% | 63.1936% |
| Fourier Amplitude Augmentation | 33.6994% | 31.6763% | 64.5885% |

CASIA 上 Strong Data Augmentation 的三个指标均为当前最优。Fourier 虽然没有取得局部最优，但相对 Baseline 仍实现 Accuracy、AUC 上升和 EER 下降。

### 4.2 MSU-MFSD

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 60.9356% | 64.0805% | 41.6813% |
| Strong Data Augmentation | 68.2259% | 60.1555% | 41.6844% |
| Appearance Randomization | 56.2080% | 56.6613% | 45.4044% |
| MixStyle | 65.6133% | 59.3304% | 42.4321% |
| Fourier Amplitude Augmentation | **75.2924%** | **65.7463%** | **39.5778%** |

MSU-MFSD 上 Fourier 同时取得最高 Accuracy、最高 AUC 和最低 EER。

### 4.3 Replay-Attack

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 48.2725% | 39.1893% | 54.5629% |
| Strong Data Augmentation | 49.1234% | 38.8743% | 54.8851% |
| Appearance Randomization | 55.5157% | 46.4997% | 51.1434% |
| MixStyle | 56.5870% | 48.3348% | 51.1476% |
| Fourier Amplitude Augmentation | **66.2087%** | **63.8448%** | **41.0995%** |

Replay-Attack 上 Fourier 的提升最明显，其中 AUC 相比 Baseline 提升超过 24 个百分点，EER 降低超过 13 个百分点。

---

## 五、三域宏平均

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER |
|---|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% |
| Strong Data Augmentation | **59.5185%** | 46.9866% | 50.7384% |
| Appearance Randomization | 50.5986% | 47.1715% | 51.7376% |
| MixStyle | 53.5927% | 46.5625% | 52.2578% |
| Fourier Amplitude Augmentation | 58.4002% | **53.7558%** | **48.4219%** |

从宏平均看，Strong 的平均 Accuracy 略高于 Fourier；Fourier 的平均 AUC 最高、平均 EER 最低。

---

## 六、三域一致性分析

相对于 Baseline，在同一个目标域上同时满足 Accuracy 上升、AUC 上升、EER 下降，才记为一次完整改善。

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 完整改进数量 |
|---|---|---|---|---:|
| Strong Data Augmentation | 是 | 否 | 否 | 1/3 |
| Appearance Randomization | 是 | 否 | 是 | 2/3 |
| MixStyle | 是 | 否 | 是 | 2/3 |
| Fourier Amplitude Augmentation | **是** | **是** | **是** | **3/3** |

Fourier 是目前唯一达到三个目标域完整一致改善的方法。

---

## 七、Fourier 相对 Baseline 的提升幅度

| 目标域 | Accuracy 提升 | AUC 提升 | EER 降低 |
|---|---:|---:|---:|
| CASIA | +5.6346 pp | +8.2756 pp | 5.3079 pp |
| MSU-MFSD | +14.3568 pp | +1.6658 pp | 2.1035 pp |
| Replay-Attack | +17.9362 pp | +24.6555 pp | 13.4634 pp |

三个目标域宏平均相比 Baseline：Accuracy +12.6425 个百分点，AUC +11.5323 个百分点，EER 降低 6.9583 个百分点。

---

## 八、消融式机制分析

### 8.1 Strong Data Augmentation

它是项目自定义的强增强配置，而非标准算法名。CASIA 上效果最好，但 MSU-MFSD 和 Replay-Attack 的 AUC/EER 没有同步改善，说明更强随机裁切、模糊和擦除既能削弱源域偏置，也可能破坏 FAS 依赖的细粒度攻击纹理。

### 8.2 Appearance Randomization

它同样是项目自定义策略。与 Strong 相比，它去掉随机尺度裁切和局部擦除，更聚焦颜色与成像外观随机化。它改善了 CASIA 和 Replay-Attack，但在 MSU-MFSD 上退化，说明不同目标域的偏移来源并不相同。

### 8.3 MixStyle

MixStyle 是正式方法名。本项目只在较浅层 `layer1/layer2` 后进行特征统计混合；结果表明它能缓解部分风格偏移，但仍不足以覆盖所有域差异。

### 8.4 Fourier Amplitude Augmentation

本项目实现的是轻量训练增强：保留源相位和大部分高频幅度，仅在同类别样本间混合低频幅度区域。当前实验结果表明，它在三域上都实现方向一致的改善，但不能据此声称低频幅度是所有 FAS 域偏移的唯一根因，也不能声称完整复现或超过某篇外部论文方法。

---

## 九、可视化内容

运行第四周分析脚本后生成：

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

最终答辩中建议优先使用 `auc_comparison.png`、`eer_comparison.png`、`improvement_heatmap.png`、`macro_auc.png` 和 `macro_eer.png`。

---

## 十、第四周结论

第四周统一对比表明，项目自定义的 Strong Data Augmentation、Appearance Randomization 和轻量集成的 MixStyle 均只能在部分目标域取得收益；本项目的轻量 Fourier Amplitude Augmentation 是当前唯一在 CASIA、MSU-MFSD 和 Replay-Attack 三个未见目标域上同时实现 Accuracy 上升、AUC 上升和 EER 下降的方案，并取得最高三域平均 AUC 和最低三域平均 EER。

但 Fourier 并未彻底解决跨域问题，尤其 CASIA 上 AUC 仍低于 0.5。最终实验结论应表述为“频域幅度增强能够稳定改善本项目设置下的跨数据集泛化”，而不能表述为“已经解决跨数据集泛化问题”。

方法复现与技术报告表述请以 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md) 为准；数据来源与预处理边界请以 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md) 为准。
