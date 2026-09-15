# 实验结果记录

本文档汇总 FAS 项目第一至第四周的正式实验结果。更详细的阶段性分析见 [`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md) 和 [`WEEK4_REPORT.md`](WEEK4_REPORT.md)。数据入口、预处理边界和质量审计流程见 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)。

> **命名说明**：`Strong Data Augmentation` 与 `Appearance Randomization` 是本项目自定义的训练策略名称，不是已有标准算法名；早期记录中的 `Strong Augmentation`、`Appearance Augmentation` 分别指同一代码配置。`MixStyle` 是已有正式方法名，本项目做轻量 ResNet18 集成；`Fourier Amplitude Augmentation` 是本项目实现的轻量频域幅度增强，而不是某篇频域 FAS 方法的完整复现。详细实现见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

## 第一周：数据准备、质量检查与单数据集 Baseline

第一周使用 ResNet18 在 OULU-NPU 上完成 Live / Spoof 二分类。数据采用自定义 Subject-disjoint 70/15/15 划分，避免同一 Subject 同时出现在 Train、Validation、Test 中。

### 1.1 数据入口与预处理边界

当前训练仓库从已经完成抽帧和人脸区域预处理的 `ProcessedData` 开始工作。样本文件名中保留了 `frame`、Subject/Client 和原视频相关信息，因此能够确认当前训练输入是处理后的图像帧。

需要明确的是：原始视频到 `ProcessedData` 的历史抽帧 FPS、Face Detector 型号和 bbox 扩展比例没有保留在当前仓库，因此本项目不补写不可验证的具体参数。该边界已在 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md) 中正式记录。

为补齐第一周的数据质量检查，仓库新增：

```text
scripts/audit_processed_data.py
scripts/run_data_audit.sh
```

执行：

```bash
bash scripts/run_data_audit.sh
```

可生成：

```text
outputs_data_audit/
├── processed_data_audit.json
└── processed_data_audit.md
```

审计内容包括图像可读性、RGB 可转换性、Live/Spoof 标签、图像尺寸、Subject/Group 数、`frame` 命名比例以及 Subject-disjoint 可划分性。该脚本为只读检查，不修改训练数据。

### 1.2 第一周正式 Baseline

| 实验编号 | 训练/测试数据集 | 模型 | 划分方式 | Accuracy | AUC | EER | EER Threshold |
|---|---|---|---|---:|---:|---:|---:|
| W1-OULU-R18-001 | OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 99.8161% | 99.9999% | 0.0526% | 0.643187 |

混淆矩阵：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

第一周结果说明模型在同域条件下具有很强判别能力，但不能据此判断跨数据集泛化能力。

### 1.3 第一周课程任务对应关系

| 课程要求 | 当前完成状态 |
|---|---|
| 准备并检查原始数据 | 已有处理后数据入口，并新增可复现质量审计 |
| 视频均匀抽帧 | 当前样本为抽帧图像，但历史抽帧参数未在仓库中保留 |
| Face Detection / Face Crop | 当前训练输入为已处理图像；检测器和 bbox 参数不可追溯 |
| 检测失败/错误裁切/异常检查 | 新增自动数据审计；历史人工裁切检查记录不可追溯 |
| Subject/Video 划分 | 已完成 Subject-disjoint 70/15/15 |
| ResNet18 Live/Spoof 训练 | 已完成 |
| Accuracy/AUC/EER/ROC/Confusion Matrix | 已完成 |
| 第一周实验记录 | 已完成并上传 GitHub |

---

## 第二周：跨数据集 Baseline

第二周固定第一周训练得到的 OULU-NPU 模型参数，直接在 CASIA、MSU-MFSD 和 Replay-Attack 上测试。目标域不参与训练、微调或参数更新。

| 实验编号 | 训练数据集 | 测试数据集 | Accuracy | AUC | EER | EER Threshold |
|---|---|---|---:|---:|---:|---:|
| W2-OULU-CASIA-001 | OULU-NPU | CASIA | 28.0648% | 23.4007% | 69.8964% | 0.666826 |
| W2-OULU-MSU-001 | OULU-NPU | MSU-MFSD | 60.9356% | 64.0805% | 41.6813% | 0.251069 |
| W2-OULU-REPLAY-001 | OULU-NPU | Replay-Attack | 48.2725% | 39.1893% | 54.5629% | 0.006060 |

与同域 Baseline 相比，三个目标域性能均显著下降，说明模型存在明显的域依赖问题。可能的域偏移来源包括摄像设备、光照、攻击介质、颜色分布、压缩和预处理差异。

---

## 第三周：泛化增强方法探索

第三周保持 ResNet18、OULU-NPU 源域训练、Subject-disjoint 划分和 Source-only 跨数据集评价协议不变，依次测试四类泛化增强策略：

1. **Strong Data Augmentation（项目自定义）**：`--augmentation strong`，更强的空间域与外观随机增强；
2. **Appearance Randomization（项目自定义）**：`--augmentation appearance`，重点随机化成像外观并保留完整人脸结构；
3. **MixStyle**：已有正式方法名，本项目插在 ResNet18 `layer1/layer2` 后；
4. **Fourier Amplitude Augmentation（项目轻量实现）**：`--method fourier`，同类别样本之间低频幅度谱混合，并保留原相位。

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

相对 Baseline，在一个目标域上同时满足 Accuracy 上升、AUC 上升和 EER 下降，记为一次“完整改善”。

| 方法 | CASIA | MSU-MFSD | Replay-Attack | 完整改善数量 |
|---|---|---|---|---:|
| Strong Data Augmentation | 是 | 否 | 否 | 1/3 |
| Appearance Randomization | 是 | 否 | 是 | 2/3 |
| MixStyle | 是 | 否 | 是 | 2/3 |
| Fourier Amplitude Augmentation | **是** | **是** | **是** | **3/3** |

Fourier 是目前唯一在三个未见目标域上均实现三项指标方向一致改善的方法。

### 3.5 Fourier 相对 Baseline 的提升

| 目标域 | Accuracy 提升 | AUC 提升 | EER 降低 |
|---|---:|---:|---:|
| CASIA | +5.6346 pp | +8.2756 pp | 5.3079 pp |
| MSU-MFSD | +14.3568 pp | +1.6658 pp | 2.1035 pp |
| Replay-Attack | +17.9362 pp | +24.6555 pp | 13.4634 pp |

三域宏平均相对 Baseline：Accuracy +12.6425 pp，AUC +11.5323 pp，EER 降低 6.9583 pp。

第三周最终推荐 Fourier Amplitude Augmentation，原因是其跨目标域改善一致性最好。需要保留实验边界：CASIA 上 Fourier 的绝对 AUC 仍低于 0.5，因此应表述为“稳定改善跨域泛化”，而不是“解决跨域泛化问题”。

详细分析见 [`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)，复现细节见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

---

## 第四周：统一对比、消融式分析与可视化

第四周不再新增训练方法，而是统一分析 Baseline、Strong Data Augmentation、Appearance Randomization、MixStyle 和 Fourier Amplitude Augmentation 五组实验。

第四周正式输出目录：[`outputs_week4/`](outputs_week4/)

主要文件：

- `week4_all_results.csv`：五种方法全部同域/跨域指标；
- `week4_macro_summary.csv`：三个目标域宏平均；
- `week4_domain_best.csv`：每个目标域各指标最优方法；
- `week4_analysis.md`：自动生成的第四周结果分析；
- `accuracy_comparison.png`：跨域 Accuracy 对比；
- `auc_comparison.png`：跨域 AUC 对比；
- `eer_comparison.png`：跨域 EER 对比；
- `macro_accuracy.png` / `macro_auc.png` / `macro_eer.png`：宏平均对比；
- `improvement_heatmap.png`：相对 Baseline 的 AUC 提升与 EER 降低热力图。

### 第四周核心结论

- Strong Data Augmentation 的三域平均 Accuracy 最高，为 59.5185%；
- Fourier 的三域平均 AUC 最高，为 53.7558%；
- Fourier 的三域平均 EER 最低，为 48.4219%；
- Fourier 是唯一在 CASIA、MSU-MFSD、Replay-Attack 三个目标域上都同时实现 Accuracy↑、AUC↑、EER↓ 的方法；
- 因此，Fourier 是当前项目跨域改善一致性最好的推荐方案。

完整第四周报告见 [`WEEK4_REPORT.md`](WEEK4_REPORT.md)。

---

## 实验产物管理建议

适合提交到 GitHub 的内容包括：

- `*.json`：训练历史、划分摘要、指标以及数据质量审计；
- `*.csv`：跨域结果汇总、宏平均、最优方法汇总；
- `*.md`：实验报告、数据预处理说明与自动分析；
- `*.png`：ROC、混淆矩阵、对比图、热力图；
- 数据审计、实验和分析脚本。

不建议提交：

- `ProcessedData/`、`RawData/` 等原始/处理后数据集；
- `*.pth`、`*.pt`、`*.ckpt` 等模型权重；
- 日志、缓存和临时文件。

当前 `.gitignore` 已排除数据集和模型权重。
