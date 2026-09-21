# 实验结果记录

本文档汇总 FAS 项目第一至第四周的正式实验结果。各阶段详细报告见：[`WEEK1_REPORT.md`](WEEK1_REPORT.md)、[`WEEK2_REPORT.md`](WEEK2_REPORT.md)、[`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md) 和 [`WEEK4_REPORT.md`](WEEK4_REPORT.md)。数据入口与预处理边界见 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)，方法实现见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

> **命名说明**：`Strong Data Augmentation` 与 `Appearance Randomization` 是本项目自定义训练策略；`MixStyle` 是已有正式方法名，本项目做 ResNet18 轻量集成；`Fourier Amplitude Augmentation` 是本项目实现的轻量频域增强，不代表完整复现某篇频域 FAS 方法。

---

## 第一周：数据准备、质量检查与单数据集 Baseline

### 1.1 ProcessedData 正式质量审计

当前训练仓库从已经完成抽帧和人脸区域预处理的 `ProcessedData` 开始工作。原始视频阶段的抽帧 FPS、Face Detector 型号和 bbox 扩展比例未保留，因此不补写不可验证的参数。

使用：

```bash
bash scripts/run_data_audit.sh
```

对实际训练输入完成正式审计：

| 数据集 | 图像数 | Live | Spoof | 无标签 | 无法读取 | 异常小图 | Group 数 | `frame` 命名比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OULU-NPU | 69,896 | 23,060 | 46,836 | 0 | 0 | 0 | 55 | 100% |
| CASIA | 11,110 | 1,810 | 9,300 | 0 | 0 | 0 | 50 | 100% |
| MSU-MFSD | 4,019 | 1,053 | 2,966 | 0 | 0 | 0 | 35 | 100% |
| Replay-Attack | 19,508 | 7,600 | 11,908 | 0 | 0 | 0 | 50 | 100% |
| **合计** | **104,533** | **33,523** | **71,010** | **0** | **0** | **0** | - | **100%** |

四个数据集全部为 `256×256` 图像。正式审计未发现无法读取、无法推断标签、无法转换 RGB 或短边低于 64 像素的异常样本。

审计产物：

```text
outputs_data_audit/
├── processed_data_audit.json
└── processed_data_audit.md
```

### 1.2 第一周正式 Baseline

OULU-NPU 使用自定义 Subject-disjoint 70/15/15 划分。

| 实验编号 | 训练/测试数据集 | 模型 | 划分方式 | Accuracy | AUC | EER | EER Threshold |
|---|---|---|---|---:|---:|---:|---:|
| W1-OULU-R18-001 | OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 99.8161% | 99.9999% | 0.0526% | 0.643187 |

混淆矩阵：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

第一周结果说明模型在 OULU-NPU 同域条件下具有很强判别能力，但不能据此判断跨数据集泛化能力。

### 1.3 第一周课程任务对应关系

| 课程要求 | 当前状态 |
|---|---|
| 准备并检查数据 | 已完成 104,533 张最终训练输入的正式质量审计 |
| 视频均匀抽帧 | `frame` 命名比例 100%，确认当前样本为抽帧图像；历史 FPS/间隔不可追溯 |
| Face Detection / Face Crop | 当前训练输入为已处理图像；历史检测器与 bbox 参数不可追溯 |
| 异常检查 | 0 损坏、0 无标签、0 异常小图、0 RGB 转换失败；历史人工裁切检查不可追溯 |
| Subject/Video 划分 | 已完成 Subject-disjoint 70/15/15 |
| ResNet18 训练 | 已完成 |
| Accuracy/AUC/EER/ROC/Confusion Matrix | 已完成 |
| 第一周实验记录 | 已完成并上传 GitHub |

详细见 [`WEEK1_REPORT.md`](WEEK1_REPORT.md)。

---

## 第二周：跨数据集 Baseline

固定第一周训练得到的 OULU-NPU 模型参数，在目标域直接测试；CASIA、MSU-MFSD、Replay-Attack 不参与训练、微调或参数更新。

| 实验编号 | 训练数据集 | 测试数据集 | 测试图像数 | Accuracy | AUC | EER | EER Threshold |
|---|---|---|---:|---:|---:|---:|---:|
| W2-OULU-CASIA-001 | OULU-NPU | CASIA | 11,110 | 28.0648% | 23.4007% | 69.8964% | 0.666826 |
| W2-OULU-MSU-001 | OULU-NPU | MSU-MFSD | 4,019 | 60.9356% | 64.0805% | 41.6813% | 0.251069 |
| W2-OULU-REPLAY-001 | OULU-NPU | Replay-Attack | 19,508 | 48.2725% | 39.1893% | 54.5629% | 0.006060 |

与同域 Baseline 相比，三个目标域性能均显著下降，证明模型存在明显域依赖。CASIA 和 Replay-Attack 的 AUC 低于 0.5，说明跨域排序能力出现严重失效。不同目标域的 EER Threshold 差异也很大，说明预测概率存在明显校准偏移。

详细见 [`WEEK2_REPORT.md`](WEEK2_REPORT.md)。

---

## 第三周：Frozen DINOv2-Reg 多源域泛化

第三周最终阶段从早期 ResNet18 数据增强探索转向 DINOv2-Reg + Multi-source DG。早期 Strong Data Augmentation、Appearance Randomization、MixStyle、Fourier Amplitude Augmentation 结果仍保留在历史报告中，但第三周最终主结果以 Frozen DINOv2-Reg 为准。

### 3.1 统一方法对比

| 方法 | CASIA Video AUC | MSU-MFSD Video AUC | Replay-Attack Video AUC | 平均 Video AUC |
|---|---:|---:|---:|---:|
| Multi-source DG + ResNet18 | 49.39% | 71.10% | 63.95% | 61.48% |
| DINOv2-Reg + Multi-source DG | 67.81% | 87.97% | 85.87% | 80.55% |
| DINOv2-Reg + SSDG-style | 75.06% | 91.29% | 79.11% | 81.82% |
| FAS-TD-SF-inspired | 72.69% | 65.47% | 73.35% | 70.50% |
| **Frozen DINOv2-Reg** | **81.32%** | **89.01%** | **82.78%** | **84.37%** |

### 3.2 第三周最终主结果

最终方法：**Frozen DINOv2-Reg Multi-source DG**

| Target | Frame AUC | Video AUC | Video EER |
|---|---:|---:|---:|
| CASIA | 80.51% | **81.32%** | 23.26% |
| MSU-MFSD | 88.22% | **89.01%** | 19.52% |
| Replay-Attack | 82.12% | **82.78%** | 27.45% |
| **Average** | - | **84.37%** | **23.41%** |

### 3.3 Tuning-depth diagnostic

| 配置 | CASIA | MSU-MFSD | Replay-Attack | 平均 Video AUC |
|---|---:|---:|---:|---:|
| DINO freeze=10 | 71.33% | 89.42% | 84.10% | 81.62% |
| DINO freeze=11 | 67.81% | 87.97% | 85.87% | 80.55% |
| DINO freeze=12 | **81.32%** | **89.01%** | 82.78% | **84.37%** |

需要注意：历史 freeze=10/11/12 运行的 backbone learning rate 并不完全一致，因此这里记录为 tuning-depth diagnostic，而不是严格控制变量的消融实验。

### 3.4 SSDG-style 参数敏感性

| Variant | CASIA | MSU-MFSD | Replay-Attack | Mean Video AUC |
|---|---:|---:|---:|---:|
| ssdg_f11_ad02_tri10 | 72.03% | 91.70% | 78.08% | 80.60% |
| ssdg_f11_ad01_tri05 | 74.99% | 91.27% | 78.33% | 81.53% |
| ssdg_f10_ad02_tri05 | 68.04% | 91.65% | 79.47% | 79.72% |
| ssdg_f12_ad02_tri05 | **83.43%** | **92.38%** | 71.14% | 82.32% |

### 3.5 融合分析

- Probability Mean：平均 Video AUC 83.26%，平均 Video EER 25.19%；
- Video Stability：平均 Video AUC 83.16%，平均 Video EER 23.70%；
- Video Reliability：平均 Video AUC 82.66%，平均 Video EER 23.59%。

融合方案没有超过 Frozen DINOv2-Reg 单模型的 84.37% 平均 Video AUC。

详细分析见 [`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)。

---

## 第四周：统一对比、消融式诊断与可视化

第四周不再训练新的第三周主模型，而是基于已有结果进行统一整理。

### 4.1 最终统一结果

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 平均 Video AUC | 平均 Video EER |
|---|---:|---:|---:|---:|---:|
| Multi-source DG + ResNet18 | 49.39% | 71.10% | 63.95% | 61.48% | 42.47% |
| DINOv2-Reg + Multi-source DG | 67.81% | 87.97% | 85.87% | 80.55% | 29.20% |
| DINOv2-Reg + SSDG-style | 75.06% | 91.29% | 79.11% | 81.82% | 26.64% |
| FAS-TD-SF-inspired | 72.69% | 65.47% | 73.35% | 70.50% | 36.72% |
| **Frozen DINOv2-Reg** | **81.32%** | 89.01% | 82.78% | **84.37%** | **23.41%** |

### 4.2 第四周分析资产

统一分析脚本：

```bash
bash scripts/run_week4_unified_analysis.sh
```

生成：

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

### 4.3 第四周核心结论

- DINOv2-Reg 是跨域性能提升的主要来源；
- 完全冻结 backbone 在当前协议下得到最高三域平均 Video AUC；
- SSDG-style 的收益具有明显 target dependence；
- Probability Mean 融合最高达到 83.26%，仍低于 Frozen DINOv2-Reg；
- Source validation AUC 接近 100% 并不代表未知域性能同样优秀；
- 最终主结果固定为 **Frozen DINOv2-Reg，Mean Video AUC = 84.37%**。

详细见 [`WEEK4_REPORT.md`](WEEK4_REPORT.md)。

---

## 实验产物管理

适合提交到 GitHub：

- `*.json`：训练历史、划分摘要、指标、数据质量审计；
- `*.csv`：跨域结果、宏平均、最优方法汇总；
- `*.md`：周报告、方法说明、数据预处理说明和自动分析；
- `*.png`：ROC、混淆矩阵、对比图、热力图；
- 数据审计、训练与分析脚本。

不提交：

- `ProcessedData/`、`RawData/`；
- `*.pth`、`*.pt`、`*.ckpt`；
- 日志、缓存和临时文件。