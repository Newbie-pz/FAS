# 第四周：实验对比、消融与结果分析

本周不再训练新的模型，而是统一分析前三周已经完成的五组方法。所有方法使用相同的 OULU-NPU 源域、相同 Subject-disjoint 划分和相同三个跨域目标数据集。

> **命名说明**：为保持图表和 CSV 简洁，本自动分析文件仍使用 `Strong / Appearance / MixStyle / Fourier` 简称。其中 `Strong` 指本项目自定义的 **Strong Data Augmentation**；`Appearance` 指本项目自定义的 **Appearance Randomization**（早期文档称 Appearance Augmentation）；`MixStyle` 是已有正式方法名，本项目做轻量 ResNet18 集成；`Fourier` 指本项目实现的轻量 **Fourier Amplitude Augmentation**，不代表完整复现某篇频域 FAS 方法。完整实现见仓库根目录 [`METHOD_IMPLEMENTATION.md`](../METHOD_IMPLEMENTATION.md)。

## 一、方法与单变量改动

| 方法简称 | 相对 Baseline 的主要改动 |
|---|---|
| Baseline | 项目基础 ResNet18 + 基础数据增强 |
| Strong | 项目自定义 Strong Data Augmentation：强空间/外观数据增强 |
| Appearance | 项目自定义 Appearance Randomization：外观/成像风格随机化 |
| MixStyle | 已有正式方法名；本项目在 ResNet18 浅层做轻量集成 |
| Fourier | 项目轻量 Fourier Amplitude Augmentation：同类样本低频幅度谱混合 |

这种设计可以从“单变量机制对照”的角度观察不同泛化策略对跨域性能的影响。技术报告第一次出现时应使用完整名称和名称性质，后续表格可继续使用简称。

## 二、同域性能对比

| 方法 | Accuracy | AUC | EER | EER Threshold |
|---|---:|---:|---:|---:|
| Baseline | 99.8161% | 0.999999 | 0.0526% | 0.643187 |
| Strong | 95.3052% | 0.999971 | 0.2104% | 0.999575 |
| Appearance | 97.4074% | 0.999624 | 1.0721% | 0.933443 |
| MixStyle | 98.5898% | 0.999914 | 0.5920% | 0.903164 |
| Fourier | 98.5373% | 0.999715 | 0.8814% | 0.794866 |

## 三、跨数据集完整对比

| 目标域 | 方法 | Accuracy | AUC | EER | ΔAccuracy | ΔAUC | EER降低量 |
|---|---|---:|---:|---:|---:|---:|---:|
| CASIA | Baseline | 28.0648% | 23.4007% | 69.8964% | +0.0000 pp | +0.0000 pp | +0.0000 pp |
| CASIA | Strong | 61.2061% | 41.9301% | 55.6456% | +33.1413 pp | +18.5294 pp | +14.2507 pp |
| CASIA | Appearance | 40.0720% | 38.3534% | 58.6650% | +12.0072 pp | +14.9526 pp | +11.2314 pp |
| CASIA | MixStyle | 38.5779% | 32.0224% | 63.1936% | +10.5131 pp | +8.6217 pp | +6.7028 pp |
| CASIA | Fourier | 33.6994% | 31.6763% | 64.5885% | +5.6346 pp | +8.2756 pp | +5.3078 pp |
| MSU-MFSD | Baseline | 60.9356% | 64.0805% | 41.6813% | +0.0000 pp | +0.0000 pp | +0.0000 pp |
| MSU-MFSD | Strong | 68.2259% | 60.1555% | 41.6844% | +7.2904 pp | -3.9251 pp | -0.0031 pp |
| MSU-MFSD | Appearance | 56.2080% | 56.6613% | 45.4044% | -4.7275 pp | -7.4192 pp | -3.7231 pp |
| MSU-MFSD | MixStyle | 65.6133% | 59.3304% | 42.4321% | +4.6778 pp | -4.7502 pp | -0.7507 pp |
| MSU-MFSD | Fourier | 75.2924% | 65.7463% | 39.5778% | +14.3568 pp | +1.6657 pp | +2.1036 pp |
| Replay-Attack | Baseline | 48.2725% | 39.1893% | 54.5629% | +0.0000 pp | +0.0000 pp | +0.0000 pp |
| Replay-Attack | Strong | 49.1234% | 38.8743% | 54.8851% | +0.8509 pp | -0.3150 pp | -0.3222 pp |
| Replay-Attack | Appearance | 55.5157% | 46.4997% | 51.1434% | +7.2432 pp | +7.3104 pp | +3.4195 pp |
| Replay-Attack | MixStyle | 56.5870% | 48.3348% | 51.1476% | +8.3145 pp | +9.1455 pp | +3.4153 pp |
| Replay-Attack | Fourier | 66.2087% | 63.8448% | 41.0995% | +17.9362 pp | +24.6555 pp | +13.4634 pp |

## 四、三个目标域宏平均

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER | 三域同时改善数 |
|---|---:|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% | - |
| Strong | 59.5185% | 46.9866% | 50.7384% | 1/3 |
| Appearance | 50.5986% | 47.1715% | 51.7376% | 2/3 |
| MixStyle | 53.5927% | 46.5625% | 52.2578% | 2/3 |
| Fourier | 58.4002% | 53.7558% | 48.4219% | 3/3 |

宏平均最优结果：

- Accuracy 最优：Strong，59.5185%。
- AUC 最优：Fourier，53.7558%。
- EER 最优：Fourier，48.4219%。

## 五、各目标域最优方法

| 目标域 | Accuracy 最优 | AUC 最优 | EER 最优 |
|---|---|---|---|
| CASIA | Strong (61.2061%) | Strong (41.9301%) | Strong (55.6456%) |
| MSU-MFSD | Fourier (75.2924%) | Fourier (65.7463%) | Fourier (39.5778%) |
| Replay-Attack | Fourier (66.2087%) | Fourier (63.8448%) | Fourier (41.0995%) |

## 六、消融式机制分析

1. **项目自定义 Strong Data Augmentation 并不等于稳定域泛化。** Strong 在 CASIA 上提升显著，但在 MSU-MFSD 和 Replay-Attack 的 AUC/EER 没有形成一致改善，说明过强裁切、模糊或遮挡可能同时破坏有用的 FAS 纹理。
2. **项目自定义 Appearance Randomization 主要缓解部分成像风格差异。** Appearance 在 Replay-Attack 上明显改善，但在 MSU-MFSD 上退化，说明颜色、亮度和清晰度随机化只能覆盖一部分域偏移。
3. **MixStyle 能改善部分浅层风格偏移，但稳定性仍不足。** 本项目在 ResNet18 `layer1/layer2` 后做轻量集成；CASIA 和 Replay-Attack 有改善，但 MSU-MFSD 的 AUC/EER 低于 Baseline。
4. **本项目轻量 Fourier Amplitude Augmentation 是当前唯一实现三域一致改善的方法。** 相比 Baseline，三个目标域均同时满足 Accuracy 上升、AUC 上升、EER 下降。
5. **同域高性能与跨域高泛化并不等价。** 所有方法在 OULU-NPU 同域仍保持较高 AUC，但跨域差异巨大，因此最终评价必须以跨域 AUC/EER 为主。

## 七、Fourier 相对 Baseline 的关键提升

| 目标域 | Accuracy 提升 | AUC 提升 | EER 降低 |
|---|---:|---:|---:|
| CASIA | +5.6346 pp | +8.2756 pp | +5.3078 pp |
| MSU-MFSD | +14.3568 pp | +1.6657 pp | +2.1036 pp |
| Replay-Attack | +17.9362 pp | +24.6555 pp | +13.4634 pp |

## 八、第四周结论

五组方法的统一对比表明，项目自定义的 Strong Data Augmentation、Appearance Randomization 和特征统计层面的 MixStyle 都只能在部分目标域取得收益，而本项目的轻量 Fourier Amplitude Augmentation 是当前唯一在 CASIA、MSU-MFSD 和 Replay-Attack 三个未见目标域上同时改善 Accuracy、AUC 与 EER 的方案。其三个目标域宏平均 AUC 最高、宏平均 EER 最低，因此将 Fourier 作为本项目当前推荐的泛化增强方法。

同时需要保留实验边界：Fourier 并未完全解决跨域问题，尤其 CASIA 的绝对 AUC 仍低于 0.5。因此最终报告应表述为“跨域泛化得到稳定改善”，而不是“已经解决跨域泛化”。
