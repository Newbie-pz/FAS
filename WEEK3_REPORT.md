# 第三周泛化增强实验报告

> **命名说明**：本报告中的 `Strong Data Augmentation` 与 `Appearance Randomization` 是本项目自定义的训练策略名称，不是具有固定公开定义的标准算法名。早期记录中的 `Strong Augmentation` 与 `Appearance Augmentation` 分别对应这两套相同代码配置。具体算子、参数和代码位置见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

> **数据入口说明**：第三周所有实验都从已经整理好的 `ProcessedData` 开始，训练前的数据来源、可追溯边界和质量审计统一见 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)。原始视频抽帧 FPS、Face Detector 型号和 bbox 扩展参数没有保留在当前仓库，因此本报告不对这些历史参数作推断性描述。

## 一、实验目标

第三周在保持 ResNet18、OULU-NPU 源数据集、Subject-disjoint 划分、训练超参数以及 Source-only 跨数据集评价协议不变的前提下，仅改变训练阶段的数据增强策略，用于研究数据增强能否降低模型对源域外观统计特征的依赖，并改善跨数据集泛化能力。

目标测试域始终为 CASIA、MSU-MFSD 和 Replay-Attack，三个目标数据集均不参与训练、微调或模型参数更新。

## 二、对比方法

本阶段共比较三种训练策略：

| 方法 | 名称性质 | 主要设置 |
|---|---|---|
| Baseline | 项目基础配置 | Resize、RandomHorizontalFlip、轻量 ColorJitter |
| Strong Data Augmentation | **项目自定义强增强配置** | RandomResizedCrop、较强 ColorJitter、RandomGrayscale、GaussianBlur、RandomErasing |
| Appearance Randomization | **项目自定义外观随机化策略** | 保持完整人脸结构，重点随机化颜色、灰度、对比度、模糊和锐度等外观/成像风格 |

### Strong Data Augmentation 的实现

代码参数：`--augmentation strong`。

核心设置：

```text
RandomResizedCrop(size=224, scale=(0.80,1.00), ratio=(0.90,1.10))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.40, contrast=0.40, saturation=0.40, hue=0.10), p=0.80
RandomGrayscale(p=0.15)
GaussianBlur(kernel_size=5, sigma=(0.1,2.0)), p=0.30
RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3))
```

该策略通过更广泛的空间与外观扰动抑制源域过拟合，但可能同时破坏 FAS 依赖的细粒度 spoof 纹理。

### Appearance Randomization 的实现

代码参数：`--augmentation appearance`。

核心设置：

```text
Resize(224,224)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.05), p=0.80
RandomGrayscale(p=0.10)
RandomAutocontrast(p=0.15)
GaussianBlur(kernel_size=3, sigma=(0.1,1.0)), p=0.15
RandomAdjustSharpness(sharpness_factor=0.6, p=0.10)
```

与 Strong 不同，它**不使用 `RandomResizedCrop` 和 `RandomErasing`**，目的是尽量保留完整人脸结构和局部纹理，只随机化颜色、对比度、灰度、模糊和锐度等成像外观因素。

## 三、同域测试结果

| 方法 | Accuracy | AUC | EER | EER Threshold |
|---|---:|---:|---:|---:|
| Baseline | 99.8161% | 99.9999% | 0.0526% | 0.643187 |
| Strong Data Augmentation | 95.3052% | 99.9971% | 0.2104% | 0.999575 |
| Appearance Randomization | 97.4074% | 99.9624% | 1.0721% | 0.933443 |

Appearance Randomization 的同域混淆矩阵为：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7335 | 295 |
| Live | 1 | 3786 |

Appearance 模型在第 7 个 epoch 触发 Early Stopping。

与 Baseline 相比，两种增强均降低了固定阈值下的源域 Accuracy，说明增强削弱了模型对 OULU-NPU 源域分布的直接拟合。与此同时，三种方法在源域上的 AUC 均接近 1，因此第三周的核心判断必须依赖跨数据集结果，而不能只看同域性能。

## 四、跨数据集正式结果

### 1. CASIA

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 28.0648% | 23.4007% | 69.8964% |
| Strong Data Augmentation | **61.2061%** | **41.9301%** | **55.6456%** |
| Appearance Randomization | 40.0720% | 38.3534% | 58.6650% |

相对 Baseline，Strong 的 Accuracy 提升 33.1413 个百分点，AUC 提升 18.5294 个百分点，EER 降低 14.2508 个百分点；Appearance 的 Accuracy 提升 12.0072 个百分点，AUC 提升 14.9527 个百分点，EER 降低 11.2314 个百分点。

因此 CASIA 上两种增强均有效，其中 Strong 效果最好。

### 2. MSU-MFSD

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 60.9356% | **64.0805%** | **41.6813%** |
| Strong Data Augmentation | **68.2259%** | 60.1555% | 41.6844% |
| Appearance Randomization | 56.2080% | 56.6613% | 45.4044% |

Strong 将固定阈值下的 Accuracy 提高 7.2903 个百分点，但 AUC 下降 3.9250 个百分点，EER 基本不变；Appearance 的 Accuracy、AUC 和 EER 均劣于 Baseline。

因此 MSU-MFSD 上不能认为增强带来了稳定泛化提升。若以阈值无关的 AUC 与 EER 为主要依据，Baseline 反而更好。

### 3. Replay-Attack

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 48.2725% | 39.1893% | 54.5629% |
| Strong Data Augmentation | 49.1234% | 38.8743% | 54.8851% |
| Appearance Randomization | **55.5157%** | **46.4997%** | **51.1434%** |

相对 Baseline，Appearance 的 Accuracy 提升 7.2432 个百分点，AUC 提升 7.3104 个百分点，EER 降低 3.4195 个百分点；Strong 基本没有改善。

因此 Replay-Attack 上 Appearance 明显优于 Strong，说明保留人脸局部纹理、主要随机化成像外观，对该目标域更有效。

## 五、三种方法总体对比

下面的宏平均仅用于描述三个目标域上的总体趋势，每个目标数据集权重相同，不代表正式标准化指标。

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER |
|---|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% |
| Strong Data Augmentation | **59.5185%** | 46.9866% | **50.7384%** |
| Appearance Randomization | 50.5986% | **47.1715%** | 51.7376% |

从宏平均看，Strong 在 Accuracy 和 EER 上整体最好，而 Appearance 的平均 AUC 略高。但这种平均值掩盖了明显的目标域差异：Strong 主要受益于 CASIA，Appearance 主要受益于 Replay-Attack，MSU-MFSD 则没有从两种增强中获得稳定收益。

## 六、结果分析

第三周最重要的发现并不是“某一种增强全面优于 Baseline”，而是不同增强策略对不同目标域具有明显的选择性。

Strong Data Augmentation 对 CASIA 最有效，说明 CASIA 与 OULU-NPU 之间的部分分布偏移可能与尺度、局部区域、颜色和清晰度差异有关。更强的随机裁切、颜色扰动、模糊和遮挡提高了源域样本的多样性，从而减少了模型对 OULU-NPU 固定外观模式的依赖。

Appearance Randomization 对 Replay-Attack 更有效，说明该目标域的差异可能更偏向成像风格、颜色、对比度、清晰度等因素。保留完整人脸结构及局部纹理，同时随机化外观风格，比强裁切和局部擦除更适合这一目标域。

MSU-MFSD 上 Baseline 的 AUC 和 EER 反而最好，说明并不是增强越强、样本变化越大就越有利于跨域泛化。某些 FAS 判别信息本身依赖细粒度纹理与攻击介质痕迹，过度改变图像可能同时破坏有用特征。

此外，多组跨域实验的 EER Threshold 与 0.5 差异很大，说明源域训练得到的预测概率在目标域上存在明显的置信度偏移和校准问题。Accuracy 使用固定 `threshold=0.5`，因此可能与 AUC、EER 给出不同结论。第三周应以 AUC、EER 与 Accuracy 联合分析，而不能只看 Accuracy。

## 七、第三周阶段性结论

第三周实验表明，训练阶段的数据增强能够在部分目标域上显著改善 ResNet18 的跨数据集泛化能力，但这种改善不具有一致性。Strong Data Augmentation 在 CASIA 上表现最好，Appearance Randomization 在 Replay-Attack 上表现最好，而 MSU-MFSD 的 AUC 与 EER 仍以 Baseline 最优。

因此，本阶段结论应表述为：**项目自定义的数据增强策略可以降低部分源域偏置，但单一增强配置无法覆盖不同 FAS 数据集之间复杂且异质的域偏移。**

后续 MixStyle 与 Fourier 实验见 [`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)。完整方法实现见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

## 八、实验规范说明

CASIA、MSU-MFSD 和 Replay-Attack 在第二、三周均作为目标测试域。后续方法探索应完整保留所有实验结果，避免根据某一目标测试集反复调参后只汇报最优配置。
