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

## 第三周：泛化增强方法探索

第三周保持 ResNet18、OULU-NPU 源域训练、Subject-disjoint 划分和 Source-only 评价协议不变，比较：

1. **Strong Data Augmentation（项目自定义）**；
2. **Appearance Randomization（项目自定义）**；
3. **MixStyle**；
4. **Fourier Amplitude Augmentation（项目轻量实现）**。

### 3.1 同域结果

| 方法 | OULU-NPU Accuracy | OULU-NPU AUC | OULU-NPU EER |
|---|---:|---:|---:|
| Baseline | 99.8161% | 99.9999% | 0.0526% |
| Strong Data Augmentation | 95.3052% | 99.9971% | 0.2104% |
| Appearance Randomization | 97.4074% | 99.9624% | 1.0721% |
| MixStyle | 98.5898% | 99.9914% | 0.5920% |
| Fourier Amplitude Augmentation | 98.5373% | 99.9715% | 0.8814% |

### 3.2 跨数据集结果

#### CASIA

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 28.0648% | 23.4007% | 69.8964% |
| Strong Data Augmentation | **61.2061%** | **41.9301%** | **55.6456%** |
| Appearance Randomization | 40.0720% | 38.3534% | 58.6650% |
| MixStyle | 38.5779% | 32.0224% | 63.1936% |
| Fourier Amplitude Augmentation | 33.6994% | 31.6763% | 64.5885% |

#### MSU-MFSD

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 60.9356% | 64.0805% | 41.6813% |
| Strong Data Augmentation | 68.2259% | 60.1555% | 41.6844% |
| Appearance Randomization | 56.2080% | 56.6613% | 45.4044% |
| MixStyle | 65.6133% | 59.3304% | 42.4321% |
| Fourier Amplitude Augmentation | **75.2924%** | **65.7463%** | **39.5778%** |

#### Replay-Attack

| 方法 | Accuracy | AUC | EER |
|---|---:|---:|---:|
| Baseline | 48.2725% | 39.1893% | 54.5629% |
| Strong Data Augmentation | 49.1234% | 38.8743% | 54.8851% |
| Appearance Randomization | 55.5157% | 46.4997% | 51.1434% |
| MixStyle | 56.5870% | 48.3348% | 51.1476% |
| Fourier Amplitude Augmentation | **66.2087%** | **63.8448%** | **41.0995%** |

### 3.3 三个目标域宏平均

| 方法 | 名称性质 | 平均 Accuracy | 平均 AUC | 平均 EER |
|---|---|---:|---:|---:|
| Baseline | 项目基础配置 | 45.7576% | 42.2235% | 55.3802% |
| Strong Data Augmentation | 项目自定义 | **59.5185%** | 46.9866% | 50.7384% |
| Appearance Randomization | 项目自定义 | 50.5986% | 47.1715% | 51.7376% |
| MixStyle | 已有正式方法名；本项目轻量集成 | 53.5927% | 46.5625% | 52.2578% |
| Fourier Amplitude Augmentation | 项目轻量频域实现 | 58.4002% | **53.7558%** | **48.4219%** |

### 3.4 三域一致性

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 完整改善数量 |
|---|---|---|---|---:|
| Strong Data Augmentation | 是 | 否 | 否 | 1/3 |
| Appearance Randomization | 是 | 否 | 是 | 2/3 |
| MixStyle | 是 | 否 | 是 | 2/3 |
| Fourier Amplitude Augmentation | **是** | **是** | **是** | **3/3** |

Fourier 是目前唯一在三个未见目标域上均同时实现 Accuracy 上升、AUC 上升和 EER 下降的方法。

### 3.5 Fourier 相对 Baseline 的提升

| 目标域 | Accuracy 提升 | AUC 提升 | EER 降低 |
|---|---:|---:|---:|
| CASIA | +5.6346 pp | +8.2756 pp | 5.3079 pp |
| MSU-MFSD | +14.3568 pp | +1.6658 pp | 2.1035 pp |
| Replay-Attack | +17.9362 pp | +24.6555 pp | 13.4634 pp |

三域宏平均相对 Baseline：Accuracy +12.6425 pp，AUC +11.5323 pp，EER 降低 6.9583 pp。

详细分析见 [`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)。

---

## 第四周：统一对比、消融式分析与可视化

第四周统一分析 Baseline、Strong Data Augmentation、Appearance Randomization、MixStyle 和 Fourier Amplitude Augmentation 五组实验。

正式输出目录：[`outputs_week4/`](outputs_week4/)

主要文件：

- `week4_all_results.csv`：五种方法全部同域/跨域指标；
- `week4_macro_summary.csv`：三个目标域宏平均；
- `week4_domain_best.csv`：各目标域最优方法；
- `week4_analysis.md`：自动结果分析；
- `accuracy_comparison.png` / `auc_comparison.png` / `eer_comparison.png`；
- `macro_accuracy.png` / `macro_auc.png` / `macro_eer.png`；
- `improvement_heatmap.png`。

第四周核心结论：

- Strong Data Augmentation 的三域平均 Accuracy 最高，为 59.5185%；
- Fourier 的三域平均 AUC 最高，为 53.7558%；
- Fourier 的三域平均 EER 最低，为 48.4219%；
- Fourier 是唯一在三个目标域上都同时实现 Accuracy↑、AUC↑、EER↓ 的方法；
- 因此 Fourier 是当前项目跨域改善一致性最好的推荐方案。

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