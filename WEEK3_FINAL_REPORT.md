# 第三周最终实验报告：跨数据集泛化增强

## 一、实验目标

第三周的目标是在保持 ResNet18、OULU-NPU 源域训练、Subject-disjoint 划分以及 Source-only 跨数据集测试协议不变的前提下，寻找能够稳定改善跨数据集泛化能力的方法。

第二周 Baseline 在三个目标域上的结果为：

| 训练数据集 | 测试数据集 | Accuracy | AUC | EER |
|---|---|---:|---:|---:|
| OULU-NPU | CASIA | 28.0648% | 23.4007% | 69.8964% |
| OULU-NPU | MSU-MFSD | 60.9356% | 64.0805% | 41.6813% |
| OULU-NPU | Replay-Attack | 48.2725% | 39.1893% | 54.5629% |

第三周依次测试了 Strong Augmentation、Appearance Augmentation、MixStyle 和 Fourier Amplitude Augmentation。

## 二、MixStyle 实验

MixStyle 在 ResNet18 的浅层特征中混合不同样本的通道均值和标准差，以扰动与成像风格相关的特征统计，同时不直接修改标签。

同域 OULU-NPU 结果：

| Accuracy | AUC | EER |
|---:|---:|---:|
| 98.5898% | 99.9914% | 0.5920% |

跨数据集结果：

| 测试数据集 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| CASIA | 38.5779% | 32.0224% | 63.1936% |
| MSU-MFSD | 65.6133% | 59.3304% | 42.4321% |
| Replay-Attack | 56.5870% | 48.3348% | 51.1476% |

相对 Baseline 的变化：

| 测试数据集 | ΔAccuracy | ΔAUC | ΔEER |
|---|---:|---:|---:|
| CASIA | +10.5131 个百分点 | +8.6217 个百分点 | -6.7028 个百分点 |
| MSU-MFSD | +4.6777 个百分点 | -4.7501 个百分点 | +0.7508 个百分点 |
| Replay-Attack | +8.3145 个百分点 | +9.1455 个百分点 | -3.4153 个百分点 |

MixStyle 在 CASIA 和 Replay-Attack 上实现了 Accuracy、AUC、EER 三项指标同步改善，但在 MSU-MFSD 上 AUC 下降且 EER 上升，因此没有达到“三个目标域一致提升”的标准。

## 三、Fourier Amplitude Augmentation 实验

Fourier 方法在训练阶段将图像转换到频域，只对低频幅度谱进行同类别样本之间的混合，并保留原样本相位信息。该设计的目的，是增强颜色、亮度、设备成像风格等低频外观变化，同时尽量保留与人脸结构和真假判别有关的内容信息。

当前实现采用同类别混合：Live 与 Live 混合，Spoof 与 Spoof 混合，避免跨类别幅度混合造成标签语义污染。

主要参数：

| 参数 | 数值 |
|---|---|
| `method` | `fourier` |
| `fourier_p` | `0.5` |
| `fourier_max_lambda` | `0.35` |
| `fourier_low_freq_ratio` | `0.10` |
| `augmentation` | `baseline` |

同域 OULU-NPU 结果：

| Accuracy | AUC | EER |
|---:|---:|---:|
| 98.5373% | 99.9715% | 0.8814% |

同域混淆矩阵：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7480 | 150 |
| Live | 17 | 3770 |

跨数据集结果：

| 测试数据集 | Accuracy | AUC | EER | EER Threshold |
|---|---:|---:|---:|---:|
| CASIA | 33.6994% | 31.6763% | 64.5885% | 0.623700 |
| MSU-MFSD | 75.2924% | 65.7463% | 39.5778% | 0.006334 |
| Replay-Attack | 66.2087% | 63.8448% | 41.0995% | 0.033450 |

### Fourier 相对 Baseline 的提升

| 测试数据集 | Baseline Accuracy | Fourier Accuracy | ΔAccuracy | Baseline AUC | Fourier AUC | ΔAUC | Baseline EER | Fourier EER | ΔEER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CASIA | 28.0648% | 33.6994% | +5.6346 个百分点 | 23.4007% | 31.6763% | +8.2756 个百分点 | 69.8964% | 64.5885% | -5.3079 个百分点 |
| MSU-MFSD | 60.9356% | 75.2924% | +14.3568 个百分点 | 64.0805% | 65.7463% | +1.6658 个百分点 | 41.6813% | 39.5778% | -2.1035 个百分点 |
| Replay-Attack | 48.2725% | 66.2087% | +17.9362 个百分点 | 39.1893% | 63.8448% | +24.6555 个百分点 | 54.5629% | 41.0995% | -13.4634 个百分点 |

Fourier 是目前第一种在 CASIA、MSU-MFSD 和 Replay-Attack 三个目标域上均同时满足以下条件的方法：

- Accuracy 上升；
- AUC 上升；
- EER 下降。

因此，从“跨目标域提升一致性”的角度，Fourier Amplitude Augmentation 是第三周目前最符合实验目标的方法。

## 四、所有方法的宏平均比较

三个目标域等权平均结果如下：

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER |
|---|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% |
| Strong Augmentation | 59.5185% | 46.9866% | 50.7384% |
| Appearance Augmentation | 50.5986% | 47.1715% | 51.7376% |
| MixStyle | 53.5927% | 46.5625% | 52.2578% |
| Fourier Amplitude Augmentation | 58.4002% | **53.7558%** | **48.4219%** |

相对 Baseline，Fourier 的三个目标域宏平均变化为：

- Accuracy：+12.6425 个百分点；
- AUC：+11.5323 个百分点；
- EER：-6.9583 个百分点。

Strong Augmentation 的平均 Accuracy 略高于 Fourier，但其 MSU-MFSD 和 Replay-Attack 的 AUC/EER 没有稳定改善。Fourier 虽然平均 Accuracy 略低于 Strong，但平均 AUC 最高、平均 EER 最低，并且是唯一在三个目标域上实现三项指标方向一致改善的方法。

## 五、最终分析

实验结果表明，跨数据集 FAS 的主要困难并非简单的源域拟合不足，而是不同数据集之间存在明显的成像风格、设备、颜色分布、压缩、攻击介质和纹理统计差异。

普通 Strong Augmentation 和 Appearance Augmentation 可以改善部分目标域，但效果具有较强的数据集依赖性；MixStyle 也能够改善 CASIA 和 Replay-Attack，但在 MSU-MFSD 上出现退化。相比之下，Fourier 低频幅度扰动在三个目标域上都表现出一致的正向变化，说明频域中的低频外观统计确实可能是导致源域依赖的重要因素之一。

需要注意的是，Fourier 虽然实现了三域一致提升，但 CASIA 上的 AUC 仍只有 31.6763%，低于 0.5，EER 仍达到 64.5885%。因此，本实验能够说明 Fourier 方法改善了跨域泛化的一致性，但不能表述为已经解决了 CASIA 上的泛化问题。对于 CASIA，Strong Augmentation 的绝对结果仍优于 Fourier。

因此，第三周最终结论建议表述为：

> 在保持相同 ResNet18 和 Source-only 评价协议的前提下，像素级强增强、外观随机化和特征统计混合都只能在部分目标域取得改善；基于低频幅度谱的 Fourier Amplitude Augmentation 则在 CASIA、MSU-MFSD 和 Replay-Attack 三个目标域上均实现 Accuracy 与 AUC 提升、EER 下降，是当前实验中跨域提升一致性最好的方法。该结果表明，针对域相关成像风格进行频域扰动，比单纯增加空间域增强强度更有利于获得稳定的跨数据集泛化收益。

## 六、第三周最终推荐方案

后续第四周的主对比建议采用以下结构：

1. Baseline：原始 ResNet18；
2. Strong Augmentation：空间域强增强；
3. Appearance Augmentation：外观风格随机化；
4. MixStyle：特征统计域泛化；
5. Fourier Amplitude Augmentation：频域域泛化，作为第三周推荐方法。

第四周重点分析 Fourier 为什么能在三个目标域上保持一致正向变化，以及不同泛化策略在不同目标域上的优势和局限。