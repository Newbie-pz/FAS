# 第三周最终实验报告：跨数据集泛化增强

> **方法命名与实现口径**：`Strong Data Augmentation` 与 `Appearance Randomization` 是本项目自定义训练策略名称，不是已有标准算法名；早期记录中的 `Strong Augmentation`、`Appearance Augmentation` 分别指同一代码配置。`MixStyle` 是已有正式方法名，本项目将其轻量集成到 ResNet18；`Fourier Amplitude Augmentation` 是本项目实现的轻量频域幅度增强，不等同于完整复现某篇频域 FAS 论文。所有算子、参数、公式、代码位置和推荐技术报告写法见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

## 一、实验目标

第三周的目标是在保持 ResNet18、OULU-NPU 源域训练、Subject-disjoint 划分以及 Source-only 跨数据集测试协议不变的前提下，寻找能够稳定改善跨数据集泛化能力的方法。

第二周 Baseline 在三个目标域上的结果为：

| 训练数据集 | 测试数据集 | Accuracy | AUC | EER |
|---|---|---:|---:|---:|
| OULU-NPU | CASIA | 28.0648% | 23.4007% | 69.8964% |
| OULU-NPU | MSU-MFSD | 60.9356% | 64.0805% | 41.6813% |
| OULU-NPU | Replay-Attack | 48.2725% | 39.1893% | 54.5629% |

第三周依次测试了：

1. **Strong Data Augmentation（项目自定义）**：`--augmentation strong`，由随机尺度裁切、较强 ColorJitter、灰度化、GaussianBlur 和 RandomErasing 组成；
2. **Appearance Randomization（项目自定义）**：`--augmentation appearance`，重点随机化颜色、对比度、灰度、自动对比度、模糊与锐度，同时不使用 RandomResizedCrop 和 RandomErasing；
3. **MixStyle**：`--method mixstyle`，插入 ResNet18 `layer1` 与 `layer2` 后，默认 `p=0.5`、`alpha=0.1`；
4. **Fourier Amplitude Augmentation（项目轻量实现）**：`--method fourier`，训练阶段对同类别样本的低频幅度谱进行混合，默认 `p=0.5`、最大混合权重 `0.35`、低频比例 `0.10`。

前两种方法是项目定义的对照配置，因此技术报告中应明确写“project-defined”；不能把名称写成某个已有论文算法。

## 二、MixStyle 实验

MixStyle 在 ResNet18 的浅层特征中混合不同样本的通道均值和标准差，以扰动与成像风格相关的特征统计，同时不直接修改标签。

本项目具体插入位置为：

```text
layer1 → MixStyle → layer2 → MixStyle → layer3 → layer4
```

默认参数为 `p=0.5`、`alpha=0.1`，混合系数服从 `Beta(0.1, 0.1)`。MixStyle 只在训练阶段启用，验证和测试阶段自动关闭。

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

### 3.1 方法性质

本项目中的 Fourier Amplitude Augmentation 是**轻量、项目自定义的频域训练增强**。其核心思想是利用 Fourier 幅度与相位分解，只扰动低频幅度统计，同时尽量保留源样本相位和其余高频信息。它不是完整复现某篇频域 FAS 方法的全部网络和损失函数。

### 3.2 具体实现

训练时首先对输入反 ImageNet Normalize，执行二维 FFT，并分解为 amplitude 与 phase。随后通过 `fftshift` 将低频移动到频谱中心，只对中心低频区域执行类内幅度混合：

```text
A_mix = (1 - λ) * A_source + λ * A_donor
P_mix = P_source
```

其中：

| 参数 | 数值 |
|---|---|
| `fourier_p` | `0.5` |
| `fourier_max_lambda` | `0.35` |
| `fourier_low_freq_ratio` | `0.10` |
| `augmentation` | `baseline` |

每个 batch 以 50% 概率执行 Fourier augmentation，`λ` 从 `[0, 0.35)` 均匀采样，只修改频谱中心约 10% 尺度的低频区域。

当前实现采用**同类别 donor**：

```text
Live  ↔ Live
Spoof ↔ Spoof
```

这样做是为了避免把可能具有标签语义的攻击纹理从 Spoof 混入 Live，或反向污染标签。验证和测试阶段不执行 Fourier augmentation。

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

Fourier 是目前第一种在 CASIA、MSU-MFSD 和 Replay-Attack 三个目标域上均同时满足 Accuracy 上升、AUC 上升和 EER 下降的方法。

## 四、所有方法的宏平均比较

三个目标域等权平均结果如下：

| 方法 | 名称性质 | 平均 Accuracy | 平均 AUC | 平均 EER |
|---|---|---:|---:|---:|
| Baseline | 项目基础配置 | 45.7576% | 42.2235% | 55.3802% |
| Strong Data Augmentation | 项目自定义 | 59.5185% | 46.9866% | 50.7384% |
| Appearance Randomization | 项目自定义 | 50.5986% | 47.1715% | 51.7376% |
| MixStyle | 已有正式方法名；本项目轻量集成 | 53.5927% | 46.5625% | 52.2578% |
| Fourier Amplitude Augmentation | 项目轻量频域实现 | 58.4002% | **53.7558%** | **48.4219%** |

相对 Baseline，Fourier 的三个目标域宏平均变化为：

- Accuracy：+12.6425 个百分点；
- AUC：+11.5323 个百分点；
- EER：-6.9583 个百分点。

Strong Data Augmentation 的平均 Accuracy 略高于 Fourier，但其 MSU-MFSD 和 Replay-Attack 的 AUC/EER 没有稳定改善。Fourier 虽然平均 Accuracy 略低于 Strong，但平均 AUC 最高、平均 EER 最低，并且是唯一在三个目标域上实现三项指标方向一致改善的方法。

## 五、最终分析

实验结果表明，跨数据集 FAS 的主要困难并非简单的源域拟合不足，而是不同数据集之间存在明显的成像风格、设备、颜色分布、压缩、攻击介质和纹理统计差异。

项目自定义的 Strong Data Augmentation 和 Appearance Randomization 可以改善部分目标域，但效果具有较强的数据集依赖性；MixStyle 也能够改善 CASIA 和 Replay-Attack，但在 MSU-MFSD 上出现退化。相比之下，Fourier 低频幅度扰动在三个目标域上都表现出一致的正向变化。

需要注意的是，Fourier 虽然实现了三域一致提升，但 CASIA 上的 AUC 仍只有 31.6763%，低于 0.5，EER 仍达到 64.5885%。因此，本实验能够说明 Fourier 方法改善了跨域泛化的一致性，但不能表述为已经解决了 CASIA 上的泛化问题。对于 CASIA，Strong Data Augmentation 的绝对结果仍优于 Fourier。

第三周最终结论建议表述为：

> 在保持相同 ResNet18 和 Source-only 评价协议的前提下，项目自定义的像素级 Strong Data Augmentation、Appearance Randomization 和特征统计层面的 MixStyle 都只能在部分目标域取得改善；本项目实现的轻量 Fourier Amplitude Augmentation 则在 CASIA、MSU-MFSD 和 Replay-Attack 三个目标域上均实现 Accuracy 与 AUC 提升、EER 下降，是当前实验中跨域提升一致性最好的方案。

## 六、第三周最终推荐方案

后续第四周主对比采用：

1. Baseline：项目基础 ResNet18；
2. Strong Data Augmentation：**项目自定义**空间域强增强；
3. Appearance Randomization：**项目自定义**外观风格随机化；
4. MixStyle：已有特征统计域泛化方法，本项目做轻量集成；
5. Fourier Amplitude Augmentation：**项目轻量实现**的频域增强，作为第三周推荐方案。

详细复现参数与代码对应关系见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。
