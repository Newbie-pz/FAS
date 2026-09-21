# 第三周最终实验报告：Frozen DINOv2-Reg 多源域泛化

> 本报告记录第三周最终封版结果。早期 ResNet18、Strong Data Augmentation、Appearance Randomization、MixStyle 与 Fourier Amplitude Augmentation 等探索仍保留在 [WEEK3_REPORT.md](WEEK3_REPORT.md) 及历史实验记录中；第三周最终主结果以本报告为准。

## 一、实验目标

第二周结果表明，单源 OULU-NPU 训练的 ResNet18 在跨数据集测试中存在明显域依赖。第三周因此转向更强的视觉基础模型与多源域泛化协议，目标是在完全不使用目标域训练数据的前提下，提高 CASIA、MSU-MFSD 和 Replay-Attack 三个未见目标域上的 Video-level AUC。

第三周最终采用：

- Backbone：DINOv2 ViT-B/14 with Registers；
- Protocol：MICO-style multi-source DG；
- 每次使用 3 个数据集作为源域，剩余 1 个作为完全未见目标域；
- Source validation 保持 group/subject-disjoint；
- 最终主方案冻结全部 12 个 DINOv2 Transformer blocks，仅训练轻量分类头；
- 目标域只用于最终测试，不参与训练、验证、模型选择或融合权重调节。

## 二、第三周方法演进

第三周先后验证了以下方向：

| 方法 | CASIA Video AUC | MSU-MFSD Video AUC | Replay-Attack Video AUC | 平均 Video AUC |
|---|---:|---:|---:|---:|
| Multi-source DG + ResNet18 | 49.39% | 71.10% | 63.95% | 61.48% |
| DINOv2-Reg + Multi-source DG | 67.81% | 87.97% | 85.87% | 80.55% |
| DINOv2-Reg + SSDG-style | 75.06% | 91.29% | 79.11% | 81.82% |
| FAS-TD-SF-inspired | 72.69% | 65.47% | 73.35% | 70.50% |
| **Frozen DINOv2-Reg** | **81.32%** | **89.01%** | **82.78%** | **84.37%** |

结果显示，单纯增加源域数量并不能保证跨域性能；引入 DINOv2-Reg 后平均 AUC 显著提升。SSDG-style 在 CASIA 和 MSU-MFSD 上继续改善，但 Replay-Attack 出现退化。最终，完全冻结 DINOv2-Reg 主干并仅训练轻量任务头取得最高三域平均 Video AUC。

## 三、最终主结果

最终第三周主模型为 **Frozen DINOv2-Reg Multi-source DG**。

| Target | Frame AUC | Video AUC | Video EER |
|---|---:|---:|---:|
| CASIA | 80.51% | **81.32%** | 23.26% |
| MSU-MFSD | 88.22% | **89.01%** | 19.52% |
| Replay-Attack | 82.12% | **82.78%** | 27.45% |
| **Average** | - | **84.37%** | **23.41%** |

三个最终 checkpoint 位于：

```text
outputs_week3_parallel_sweep/dino_f12_lr5e6_train/CASIA/best.pth
outputs_week3_parallel_sweep/dino_f12_lr5e6_train/MSU-MFSD/best.pth
outputs_week3_parallel_sweep/dino_f12_lr5e6_train/Replay-Attack/best.pth
```

> Checkpoint 文件体积较大，受 `.gitignore` 管理，不提交 GitHub。

## 四、冻结策略诊断

第三周参数扫描中观察到：

| 配置 | CASIA | MSU-MFSD | Replay-Attack | 平均 Video AUC |
|---|---:|---:|---:|---:|
| DINO freeze=10 | 71.33% | 89.42% | 84.10% | 81.62% |
| DINO freeze=11 | 67.81% | 87.97% | 85.87% | 80.55% |
| DINO freeze=12 | **81.32%** | **89.01%** | 82.78% | **84.37%** |

完全冻结 DINOv2 主干后，整体未知域性能最好。与此同时，多个可训练配置的 Source-Worst AUC 已接近 100%，但未知目标域平均 AUC 仍明显较低。这说明更强的源域拟合并不等价于更好的跨域泛化。

需要注意：历史 freeze=10/11/12 实验的 backbone learning rate 并非完全一致，因此该结果应表述为 **tuning-depth diagnostic**，而不是严格控制变量的消融实验。

## 五、SSDG-style 分析

代表性参数扫描结果：

| Variant | CASIA | MSU-MFSD | Replay-Attack | Mean Video AUC |
|---|---:|---:|---:|---:|
| ssdg_f11_ad02_tri10 | 72.03% | 91.70% | 78.08% | 80.60% |
| ssdg_f11_ad01_tri05 | 74.99% | 91.27% | 78.33% | 81.53% |
| ssdg_f10_ad02_tri05 | 68.04% | 91.65% | 79.47% | 79.72% |
| ssdg_f12_ad02_tri05 | **83.43%** | **92.38%** | 71.14% | 82.32% |

Frozen SSDG 在 CASIA 与 MSU-MFSD 上取得很高 AUC，但 Replay-Attack 明显下降，说明该类域对抗与特征约束对不同目标域的收益并不稳定。

本项目实现应称为 **SSDG-style**：它保留单侧真实样本域对抗和非对称 triplet grouping 等核心思想，但骨干替换为 DINOv2-Reg，因此不是原论文网络的逐项复现。

## 六、融合实验

DINOv2-Reg 与 SSDG-style 之间存在一定互补性，因此进一步测试固定概率平均、rank fusion、置信度加权、稳定性加权和可靠性融合。

主要结果：

- 固定 0.5/0.5 Probability Mean：平均 Video AUC **83.26%**；
- Video Stability Fusion：平均 Video AUC **83.16%**；
- Video Reliability Fusion：平均 Video AUC 82.66%，平均 Video EER 23.59%。

融合可以改善部分指标，但均未超过 Frozen DINOv2-Reg 单模型的 **84.37%** 平均 Video AUC，因此第三周最终仍采用 Frozen DINOv2-Reg 作为主方案。

## 七、第三周最终结论

第三周最终结果表明：

1. 多源训练本身不足以解决跨数据集 FAS；
2. DINOv2-Reg 显著增强了跨域表征能力；
3. SSDG-style 可改善部分目标域，但存在明显目标域敏感性；
4. 在当前数据和协议下，完全冻结 DINOv2-Reg 主干比进一步微调高层 Transformer blocks 更稳定；
5. 最终主模型在 CASIA、MSU-MFSD、Replay-Attack 上分别取得 **81.32%、89.01%、82.78%** Video AUC，三域平均达到 **84.37%**。

因此，第三周正式封版结果为：

> **Frozen DINOv2-Reg Multi-source DG，Mean Video AUC = 84.37%。**

第四周不再修改该主结果，而是围绕已有实验进行统一对比、消融式诊断和可视化分析，详见 [WEEK4_REPORT.md](WEEK4_REPORT.md)。
