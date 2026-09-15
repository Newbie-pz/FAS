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

## 八、第三周：泛化增强方法

第三周在相同骨干、源数据和评价协议下测试四种泛化增强策略。

### 1. Strong Data Augmentation（项目自定义）

代码：`--augmentation strong`。组合随机尺度裁切、较强颜色扰动、灰度化、模糊和 Random Erasing。CASIA 提升明显，但 MSU-MFSD 和 Replay-Attack 的 AUC/EER 改善不稳定。

### 2. Appearance Randomization（项目自定义；旧称 Appearance Augmentation）

代码：`--augmentation appearance`。重点随机化颜色、亮度、对比度、灰度、自动对比度、模糊和锐度，同时不使用 RandomResizedCrop 和 RandomErasing。CASIA 和 Replay-Attack 有改善，但 MSU-MFSD 退化。

### 3. MixStyle

本项目在 ResNet18 `layer1` 和 `layer2` 后插入 MixStyle，以 `p=0.5` 的概率混合 mini-batch 内样本的通道均值和标准差，混合系数服从 `Beta(0.1, 0.1)`；验证和测试阶段关闭。

### 4. Fourier Amplitude Augmentation（项目轻量实现）

训练阶段在同类别样本之间混合低频幅度谱并保留源相位。默认 `p=0.5`、最大混合权重 `0.35`、低频区域比例 `0.10`。

详细实现见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

### 第三周宏平均结果

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER | 三域完整改善数 |
|---|---:|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% | - |
| Strong Data Augmentation | **59.5185%** | 46.9866% | 50.7384% | 1/3 |
| Appearance Randomization | 50.5986% | 47.1715% | 51.7376% | 2/3 |
| MixStyle | 53.5927% | 46.5625% | 52.2578% | 2/3 |
| Fourier Amplitude Augmentation | 58.4002% | **53.7558%** | **48.4219%** | **3/3** |

Fourier 是目前唯一在 CASIA、MSU-MFSD、Replay-Attack 三个目标域上均同时实现 Accuracy 上升、AUC 上升、EER 下降的方法。

第三周报告：[`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)

## 九、第四周：统一对比、消融式分析与可视化

第四周统一分析：Baseline、Strong Data Augmentation、Appearance Randomization、MixStyle、Fourier Amplitude Augmentation。

运行：

```bash
bash scripts/run_week4_analysis.sh
```

输出目录：[`outputs_week4/`](outputs_week4/)

核心产物：

```text
week4_all_results.csv
week4_macro_summary.csv
week4_domain_best.csv
week4_analysis.md
accuracy_comparison.png
auc_comparison.png
eer_comparison.png
macro_accuracy.png
macro_auc.png
macro_eer.png
improvement_heatmap.png
```

核心结论：

- Strong Data Augmentation 的三域平均 Accuracy 最高；
- Fourier 的三域平均 AUC 最高；
- Fourier 的三域平均 EER 最低；
- Fourier 是唯一三域都实现 Accuracy↑、AUC↑、EER↓ 的方法；
- CASIA 上 Fourier 的绝对 AUC 仍低于 0.5，因此应表述为“稳定改善跨域泛化”，不能表述为“已经解决跨域泛化”。

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
