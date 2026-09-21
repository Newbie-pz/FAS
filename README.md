# FAS：跨数据集虚假人脸检测与域泛化实验

本项目用于完成虚假人脸检测（Face Anti-Spoofing, FAS）课程实践。当前以单张 RGB 人脸图像为输入，使用 ResNet18 完成 Live / Spoof 二分类，并围绕“同域高性能但跨域泛化不足”的问题，逐步开展数据质量审计、跨数据集测试、数据增强、特征统计域泛化、频域增强以及统一实验分析。

项目第一至第四周实验已经完成，第五周进入最终技术报告与答辩整理阶段。

> **方法命名说明**：`Strong Data Augmentation` 与 `Appearance Randomization` 是本项目定义的训练策略名称，并非具有唯一标准定义的公开算法名；`MixStyle` 是已有正式方法名；本项目的 `Fourier Amplitude Augmentation` 是轻量、项目自定义的频域增强实现，不等同于完整复现某篇频域 FAS 论文。实现细节见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

## 一、任务定义

标签约定：

- `Live = 1`：真实活体人脸；
- `Spoof = 0`：打印照片、屏幕回放等呈现攻击。

模型输入为处理后的人脸 RGB 图像，输出 Live / Spoof 二分类结果。统一使用 Live 类概率计算 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix。

## 二、数据集与正式质量审计

默认数据根目录：

```text
/root/Desktop/code/FAS/ProcessedData
```

当前使用：

```text
ProcessedData/
├── CASIA/
├── MSU-MFSD/
├── OULU-NPU/
└── Replay-Attack/
```

数据集目录通过 `.gitignore` 排除，不上传 GitHub。

### 数据预处理边界

当前仓库从已经完成抽帧和人脸区域预处理的 `ProcessedData` 开始训练与评估。仓库能够完整复现标签读取、Subject/Client 解析、Subject-disjoint 划分、模型训练、跨域测试和域泛化实验；但原始视频阶段的抽帧 FPS、Face Detector 型号和 bbox 扩展比例没有保留，因此技术报告中不对这些不可追溯信息作推断性描述。

完整说明见 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)。

### ProcessedData 正式审计结果

运行：

```bash
bash scripts/run_data_audit.sh
```

正式审计覆盖 **104,533 张图像**：

| 数据集 | 图像数 | Live | Spoof | 无标签 | 无法读取 | 异常小图 | Group 数 | `frame` 命名比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OULU-NPU | 69,896 | 23,060 | 46,836 | 0 | 0 | 0 | 55 | 100% |
| CASIA | 11,110 | 1,810 | 9,300 | 0 | 0 | 0 | 50 | 100% |
| MSU-MFSD | 4,019 | 1,053 | 2,966 | 0 | 0 | 0 | 35 | 100% |
| Replay-Attack | 19,508 | 7,600 | 11,908 | 0 | 0 | 0 | 50 | 100% |
| **合计** | **104,533** | **33,523** | **71,010** | **0** | **0** | **0** | - | **100%** |

四个数据集全部为 `256×256` 图像，未发现无法读取、无法推断标签、无法转换 RGB 或短边低于 64 像素的样本。

审计结果：[`outputs_data_audit/processed_data_audit.md`](outputs_data_audit/processed_data_audit.md)

## 三、数据划分与泄漏控制

为避免同一 Subject/Client 的高度相似帧同时进入 Train 和 Test，训练阶段采用 Subject-disjoint 划分，而不是帧级随机划分。

默认比例：

| 参数 | 数值 |
|---|---:|
| `train_ratio` | `0.70` |
| `val_ratio` | `0.15` |
| `test_ratio` | `0.15` |

代码保证：

```text
Train Subjects ∩ Validation Subjects = ∅
Train Subjects ∩ Test Subjects       = ∅
Validation Subjects ∩ Test Subjects  = ∅
```

当前属于本项目自定义 Subject-disjoint 70/15/15 划分，不应表述为各数据集官方 Protocol。

## 四、Baseline 模型

Baseline 使用 torchvision ResNet18，默认加载 ImageNet 预训练权重，并将最后分类层替换为 2 类输出。

| 参数 | 数值 |
|---|---|
| `image_size` | `224` |
| `epochs` | `20` |
| `batch_size` | `64` |
| `lr` | `1e-4` |
| `weight_decay` | `1e-4` |
| `patience` | `5` |
| `optimizer` | `AdamW` |
| `pretrained` | `ImageNet` |

## 五、评价指标

项目统一报告：

- Accuracy：固定阈值 `0.5` 下的分类正确率；
- AUC：基于 Live 类概率计算的 ROC 曲线下面积；
- EER：错误接受率与错误拒绝率相等附近的错误率；
- ROC Curve；
- Confusion Matrix。

跨域分析中重点结合 AUC 和 EER 判断模型排序能力和跨域错误率，而不只观察固定阈值 Accuracy。

## 六、第一周：数据审计与 OULU-NPU 同域 Baseline

第一周完成：

1. 对最终训练输入进行 104,533 张图像的正式质量审计；
2. 按 Subject-disjoint 70/15/15 建立 Train/Validation/Test；
3. 使用 ResNet18 完成 Live/Spoof 二分类训练；
4. 输出 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix；
5. 保存训练历史、划分摘要和正式实验记录。

原始视频均匀抽帧与 Face Detector / Face Crop 的历史具体参数没有保留；当前数据的 `frame` 命名比例为 100%，能够确认训练输入为抽帧后的处理图像，但不能反推出历史 FPS 或检测器型号。

正式结果：

| 数据集 | 模型 | 划分方式 | Accuracy | AUC | EER |
|---|---|---|---:|---:|---:|
| OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 99.8161% | 99.9999% | 0.0526% |

第一周报告：[`WEEK1_REPORT.md`](WEEK1_REPORT.md)

第一周正式产物：

```text
outputs/OULU-NPU/
├── history.json
├── split_summary.json
├── test_metrics.json
├── test_roc.png
└── test_confusion_matrix.png
```

## 七、第二周：跨数据集 Baseline

固定 OULU-NPU 训练模型，目标域仅用于测试，不参与训练或微调。

| 训练数据集 | 测试数据集 | 测试图像数 | Accuracy | AUC | EER |
|---|---|---:|---:|---:|---:|
| OULU-NPU | CASIA | 11,110 | 28.0648% | 23.4007% | 69.8964% |
| OULU-NPU | MSU-MFSD | 4,019 | 60.9356% | 64.0805% | 41.6813% |
| OULU-NPU | Replay-Attack | 19,508 | 48.2725% | 39.1893% | 54.5629% |

跨域性能大幅下降，证明 Baseline 存在明显域依赖。

运行：

```bash
bash scripts/run_week2_cross_dataset.sh
```

第二周报告：[`WEEK2_REPORT.md`](WEEK2_REPORT.md)

## 八、第三周：Frozen DINOv2-Reg 多源域泛化

第三周在早期 ResNet18 数据增强探索基础上，转向 DINOv2-Reg + Multi-source DG 高性能路线。

统一采用 3 个源域训练、1 个完全未见目标域测试的 MICO-style DG 设置。目标域不参与训练、验证、模型选择或融合权重调节。

主要结果：

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 平均 Video AUC |
|---|---:|---:|---:|---:|
| Multi-source DG + ResNet18 | 49.39% | 71.10% | 63.95% | 61.48% |
| DINOv2-Reg + Multi-source DG | 67.81% | 87.97% | 85.87% | 80.55% |
| DINOv2-Reg + SSDG-style | 75.06% | 91.29% | 79.11% | 81.82% |
| FAS-TD-SF-inspired | 72.69% | 65.47% | 73.35% | 70.50% |
| **Frozen DINOv2-Reg** | **81.32%** | **89.01%** | **82.78%** | **84.37%** |

最终第三周主模型为 **Frozen DINOv2-Reg Multi-source DG**，平均 Video AUC 为 **84.37%**，平均 Video EER 为 **23.41%**。

冻结深度诊断显示，完全冻结 12 个 DINOv2 Transformer blocks 比解冻高层 block 的配置更稳定。多个可训练配置的 Source validation AUC 接近 100%，但未知域性能反而更低，说明源域拟合能力与跨域泛化能力并不等价。

第三周最终报告：[`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)

> 早期 Strong Data Augmentation、Appearance Randomization、MixStyle、Fourier Amplitude Augmentation 等 ResNet18 探索仍保留在 [`WEEK3_REPORT.md`](WEEK3_REPORT.md) 和历史实验记录中，但不再作为第三周最终主结果。

## 九、第四周：统一对比、消融式诊断与可视化

第四周不再继续训练新的第三周主模型，而是围绕现有结果进行统一整理和分析，包括：

- 统一方法对比；
- DINOv2 tuning-depth diagnostic；
- SSDG-style 参数敏感性；
- DINO + SSDG 融合策略分析；
- Source validation 与 unseen target 泛化差距；
- 最终模型 Video-level ROC；
- CASIA / MSU-MFSD / Replay-Attack 混淆矩阵。

运行：

```bash
bash scripts/run_week4_unified_analysis.sh
```

输出目录：

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

第四周统一结果：

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 平均 Video AUC | 平均 Video EER |
|---|---:|---:|---:|---:|---:|
| Multi-source DG + ResNet18 | 49.39% | 71.10% | 63.95% | 61.48% | 42.47% |
| DINOv2-Reg + Multi-source DG | 67.81% | 87.97% | 85.87% | 80.55% | 29.20% |
| DINOv2-Reg + SSDG-style | 75.06% | 91.29% | 79.11% | 81.82% | 26.64% |
| FAS-TD-SF-inspired | 72.69% | 65.47% | 73.35% | 70.50% | 36.72% |
| **Frozen DINOv2-Reg** | **81.32%** | 89.01% | 82.78% | **84.37%** | **23.41%** |

核心结论：

- DINOv2-Reg 相比 ResNet18 显著提升跨域性能；
- 完全冻结 DINOv2 主干在当前协议下得到最佳三域平均 AUC；
- SSDG-style 在 CASIA/MSU-MFSD 上有效，但 Replay-Attack 退化明显；
- 概率融合最高平均 AUC 为 83.26%，仍低于 Frozen DINOv2-Reg；
- Source validation 接近 100% 并不代表未知域性能同样优秀。

第四周报告：[`WEEK4_REPORT.md`](WEEK4_REPORT.md)

## 十、项目结构

```text
FAS/
├── README.md
├── DATA_PREPROCESSING.md
├── METHOD_IMPLEMENTATION.md
├── EXPERIMENT_RESULTS.md
├── WEEK1_REPORT.md
├── WEEK2_REPORT.md
├── WEEK3_REPORT.md
├── WEEK3_FINAL_REPORT.md
├── WEEK4_REPORT.md
├── FAS_Chinese_Experiment_Report.docx
├── FAS_Defense_Presentation_CN.pptx
├── datasets.py
├── model.py
├── dg_methods.py
├── metrics.py
├── train.py
├── evaluate_cross_dataset.py
├── summarize_cross_results.py
├── compare_week3_results.py
├── analyze_week4.py
├── scripts/
│   ├── audit_processed_data.py
│   ├── run_data_audit.sh
│   ├── run_week2_cross_dataset.sh
│   ├── run_week3_generalization.sh
│   ├── run_week3_appearance.sh
│   ├── run_week3_mixstyle.sh
│   ├── run_week3_fourier.sh
│   ├── run_week3_extra_methods.sh
│   └── run_week4_analysis.sh
├── outputs_data_audit/
├── outputs/
├── outputs_cross/
├── outputs_week3/
├── outputs_week3_cross/
├── outputs_week4/
├── requirements.txt
└── .gitignore
```

## 十一、实验产物与 GitHub 管理

推荐提交：

- `*.json`：指标、训练历史、划分摘要、数据质量审计；
- `*.csv`：实验结果汇总；
- `*.md`：周报告、方法说明、数据说明和自动分析；
- `*.png`：ROC、混淆矩阵、对比图和热力图；
- 实验报告 Word、答辩 PPT；
- 数据审计、训练与分析脚本。

不提交：

- `ProcessedData/`、`RawData/`；
- `*.pth`、`*.pt`、`*.ckpt`；
- 日志、缓存、临时文件。

## 十二、运行环境

| 参数 | 数值 |
|---|---|
| Python | `3.10` |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0` |
| CUDA | `12.8` |
| GPU | `NVIDIA GeForce RTX 4090 D` |

## 十三、文档入口

- 数据预处理与质量审计：[`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)
- 方法命名与实现细节：[`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)
- 完整结果总表：[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)
- 第一周报告：[`WEEK1_REPORT.md`](WEEK1_REPORT.md)
- 第二周报告：[`WEEK2_REPORT.md`](WEEK2_REPORT.md)
- 第三周最终报告：[`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)
- 第四周完整报告：[`WEEK4_REPORT.md`](WEEK4_REPORT.md)
