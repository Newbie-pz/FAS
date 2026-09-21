# 第四周实验报告：统一对比、消融式分析与可视化

> 第四周不再继续训练新的第三周主模型，而是基于已有实验结果进行统一比较、参数敏感性分析、融合分析、Source-Target 泛化差距分析及可视化。第三周正式主结果固定为 **Frozen DINOv2-Reg Multi-source DG，Mean Video AUC = 84.37%**。

## 一、实验目的

第四周重点回答以下问题：

1. 强 backbone 对跨域 FAS 的实际贡献有多大；
2. DINOv2 微调深度与未知域泛化之间是什么关系；
3. SSDG-style 域泛化约束是否在不同目标域上稳定；
4. 多模型融合是否能够超过最佳单模型；
5. Source validation 接近 100% 时，为什么未知目标域仍存在明显性能下降；
6. 如何通过 ROC、混淆矩阵和对比图直观展示上述现象。

所有结果均来自已完成实验，第四周分析脚本不重新训练模型。

## 二、统一方法对比

| 方法 | CASIA AUC | MSU-MFSD AUC | Replay-Attack AUC | 平均 AUC | 平均 EER |
|---|---:|---:|---:|---:|---:|
| Multi-source DG + ResNet18 | 49.39% | 71.10% | 63.95% | 61.48% | 42.47% |
| DINOv2-Reg + Multi-source DG | 67.81% | 87.97% | **85.87%** | 80.55% | 29.20% |
| DINOv2-Reg + SSDG-style | 75.06% | **91.29%** | 79.11% | 81.82% | 26.64% |
| FAS-TD-SF-inspired | 72.69% | 65.47% | 73.35% | 70.50% | 36.72% |
| **Frozen DINOv2-Reg** | **81.32%** | 89.01% | 82.78% | **84.37%** | **23.41%** |

统一结果表明：

- ResNet18 多源训练平均 Video AUC 仅为 61.48%；
- 换用 DINOv2-Reg 后提高至 80.55%，提升约 19.07 个百分点；
- SSDG-style 进一步提高 CASIA/MSU-MFSD，但 Replay-Attack 退化；
- Frozen DINOv2-Reg 获得最高三域平均 AUC 84.37%，同时平均 EER 最低，为 23.41%。

因此，第四周统一分析仍以 Frozen DINOv2-Reg 为最终主模型。

## 三、DINOv2 微调深度诊断

代表性结果：

| Tuning depth | CASIA | MSU-MFSD | Replay-Attack | Mean Video AUC |
|---|---:|---:|---:|---:|
| freeze=10 | 71.33% | 89.42% | 84.10% | 81.62% |
| freeze=11 | 67.81% | 87.97% | 85.87% | 80.55% |
| freeze=12 | **81.32%** | **89.01%** | 82.78% | **84.37%** |

在当前实验中，完全冻结 DINOv2 主干得到最好的平均未知域性能。与此同时，freeze=10 等配置的 Source-Worst AUC 可以达到约 100%，但目标域平均 AUC 反而下降，说明源域拟合能力与跨域泛化能力并不等价。

该结果更适合作为 **tuning-depth diagnostic**，而不是严格控制变量消融，因为历史实验的 backbone learning rate 并非完全相同。

## 四、SSDG-style 参数敏感性

| Variant | CASIA | MSU-MFSD | Replay-Attack | Mean Video AUC |
|---|---:|---:|---:|---:|
| ssdg_f11_ad02_tri10 | 72.03% | 91.70% | 78.08% | 80.60% |
| ssdg_f11_ad01_tri05 | 74.99% | 91.27% | 78.33% | 81.53% |
| ssdg_f10_ad02_tri05 | 68.04% | 91.65% | 79.47% | 79.72% |
| ssdg_f12_ad02_tri05 | **83.43%** | **92.38%** | 71.14% | 82.32% |

其中 frozen SSDG 在 CASIA 和 MSU-MFSD 上表现很强，但 Replay-Attack 仅有 71.14% AUC。这表明域对抗与非对称特征约束存在明显的 target-dependent behavior。

因此，SSDG-style 更适合作为第四周的参数敏感性和 failure-case 分析，而不是替代最终 Frozen DINOv2-Reg 主模型。

## 五、模型融合分析

DINOv2-Reg 与 SSDG-style 在不同目标域上具有一定互补性，因此测试了多种 label-free 融合方案。

| Fusion | Mean Video AUC | Mean Video EER |
|---|---:|---:|
| Probability Mean | **83.26%** | 25.19% |
| Logit Mean | 82.84% | 25.88% |
| Video Confidence | 82.83% | 25.20% |
| Video Stability | 83.16% | 23.70% |
| Video Entropy-Stability | 81.99% | 24.51% |
| Video Reliability | 82.66% | **23.59%** |

固定概率平均在融合方案中获得最高平均 AUC，但仍低于 Frozen DINOv2-Reg 单模型的 84.37%。Video Reliability 可以进一步降低 EER，但没有同步提高 AUC。

因此，融合结果用于说明模型间互补性和校准差异，不替代第三周主结果。

## 六、Source-Target 泛化差距

多个配置在源域验证上的 AUC 接近 100%，但未知目标域平均 AUC 仍只有约 80%左右。例如：

- dino_f12_lr5e6：Mean Source-Worst AUC 99.74%，Mean Target Video AUC 84.37%；
- dino_f10_lr2e6：Mean Source-Worst AUC 约 100%，Mean Target Video AUC 81.62%；
- 多个 SSDG-style 配置：Source-Worst AUC 约 100%，但 Target Mean AUC 79%～82%。

这说明模型可能在源域中利用摄像头、压缩噪声、背景结构、颜色分布或攻击设备特征完成高质量分类，而这些域特定线索在新的数据集上并不稳定。

Frozen DINOv2-Reg 并没有追求最高 Source validation AUC，却获得最高未知域平均 AUC，进一步支持“保持基础模型通用表征、减少源域过拟合”的结论。

## 七、可视化资产

运行：

```bash
bash scripts/run_week4_unified_analysis.sh
```

会生成：

```text
outputs_week4_analysis/
├── week4_summary.md
├── week4_method_comparison.csv
├── 01_method_comparison_video_auc.png
├── 02_method_mean_video_auc.png
├── 03_tuning_depth_comparison.png
├── 04_ssdg_parameter_sensitivity.png
├── 05_fusion_mean_video_auc.png
├── 06_source_target_generalization_gap.png
├── 06_final_model_video_roc.png
├── 07_confusion_CASIA.png
├── 07_confusion_MSU-MFSD.png
└── 07_confusion_Replay-Attack.png
```

推荐答辩优先使用：

1. `01_method_comparison_video_auc.png`：三域方法对比；
2. `02_method_mean_video_auc.png`：平均 AUC；
3. `03_tuning_depth_comparison.png`：DINOv2 冻结深度；
4. `06_source_target_generalization_gap.png`：源域/未知域差距；
5. `06_final_model_video_roc.png`：最终模型 ROC。

混淆矩阵、SSDG 参数敏感性和融合策略图可以作为补充材料。

## 八、最终结果

第四周最终确认第三周主模型不再修改：

| Target | Video AUC |
|---|---:|
| CASIA | 81.32% |
| MSU-MFSD | 89.01% |
| Replay-Attack | 82.78% |
| **Average** | **84.37%** |

最终模型为：

> **Frozen DINOv2-Reg Multi-source DG**

对应平均 Video EER 为 **23.41%**。

## 九、第四周结论

第四周统一分析表明：

1. 强视觉基础模型是跨域 FAS 性能提升的主要来源；
2. 多源训练本身并不足以获得稳定的未知域泛化；
3. 过度微调 DINOv2 高层 Transformer blocks 可能加重源域特定表征适应；
4. SSDG-style 在部分目标域有效，但目标域敏感性明显；
5. 多模型融合具有互补性，但没有超过 Frozen DINOv2-Reg 单模型；
6. Source validation 接近 100% 并不意味着未知域性能同样优秀；
7. 当前最稳定方案是保留 DINOv2-Reg 的预训练表示，只训练轻量任务头。

因此，本项目第四周的最终结论为：

> **在当前 MICO-style multi-source DG 设置下，Frozen DINOv2-Reg 在三个未见目标域上取得 84.37% 的平均 Video AUC，是目前整体性能最稳定的方案。**
