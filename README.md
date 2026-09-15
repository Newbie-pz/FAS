# FAS：跨数据集虚假人脸检测与域泛化实验

本项目用于完成虚假人脸检测（Face Anti-Spoofing, FAS）课程实践。当前以单张 RGB 人脸图像为输入，使用 ResNet18 完成 Live / Spoof 二分类，并围绕“同域高性能但跨域泛化不足”的问题，逐步开展跨数据集测试、数据增强、特征统计域泛化、频域增强以及统一消融式分析。

项目当前已经完成第一至第四周实验，进入最终整理与答辩阶段。

> **方法命名说明**：`Strong Data Augmentation` 与 `Appearance Randomization` 是本项目为了组织对照实验而定义的训练策略名称，并非具有唯一标准定义的公开算法名；`MixStyle` 是已有正式方法名；本项目的 `Fourier Amplitude Augmentation` 是轻量、项目自定义的频域增强实现，不等同于完整复现某篇频域 FAS 论文。所有方法的具体算子、参数、作用位置和推荐技术报告表述见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

## 一、任务定义

标签约定：

- `Live = 1`：真实活体人脸；
- `Spoof = 0`：打印照片、屏幕回放等呈现攻击。

模型输入为裁切后的 RGB 人脸图像，输出 Live / Spoof 二分类结果。统一使用 Live 类概率计算 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix。

## 二、数据集

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

数据集目录已通过 `.gitignore` 排除，不上传到 GitHub。

## 三、数据划分与泄漏控制

为避免同一视频相邻帧跨集合造成数据泄漏，训练阶段采用 Subject-disjoint 划分，而不是帧级随机划分。

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

当前属于自定义 Subject-disjoint 70/15/15 划分，不应表述为各数据集官方 Protocol。

## 四、Baseline 模型

Baseline 使用 torchvision 的 ResNet18，默认加载 ImageNet 预训练权重，并将最后的分类层替换为 2 类输出。

基础训练参数：

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

跨域分析中不只看 Accuracy，而重点结合 AUC 和 EER 判断模型排序能力和跨域错误率。

## 六、第一周：OULU-NPU 同域 Baseline

| 数据集 | 模型 | 划分方式 | Accuracy | AUC | EER |
|---|---|---|---:|---:|---:|
| OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 99.8161% | 99.9999% | 0.0526% |

结果表明模型在 OULU-NPU 同域条件下几乎完全正确，但这不能代表模型具有强跨数据集泛化能力。

## 七、第二周：跨数据集 Baseline

固定 OULU-NPU 训练得到的模型参数，目标域仅用于测试，不参与训练或微调。

| 训练数据集 | 测试数据集 | Accuracy | AUC | EER |
|---|---|---:|---:|---:|
| OULU-NPU | CASIA | 28.0648% | 23.4007% | 69.8964% |
| OULU-NPU | MSU-MFSD | 60.9356% | 64.0805% | 41.6813% |
| OULU-NPU | Replay-Attack | 48.2725% | 39.1893% | 54.5629% |

跨域性能大幅下降，说明 Baseline 存在明显域依赖。

运行：

```bash
bash scripts/run_week2_cross_dataset.sh
```

## 八、第三周：泛化增强方法

第三周在相同骨干、源数据和评价协议下，测试四种泛化增强策略。为避免误解，以下名称性质不同：前两种是**本项目定义的训练配置**，MixStyle 是已有正式方法名，Fourier 是本项目实现的轻量频域增强。

### 1. Strong Data Augmentation（项目自定义）

代码中对应 `--augmentation strong`。该配置组合随机尺度裁切、较强颜色扰动、灰度化、模糊和 Random Erasing，用于扩大源域的空间与外观变化。它不是某篇论文中具有固定定义的独立算法。

CASIA 提升明显，但 MSU-MFSD 和 Replay-Attack 的 AUC/EER 改善不稳定。

运行：

```bash
bash scripts/run_week3_generalization.sh
```

### 2. Appearance Randomization（项目自定义；旧称 Appearance Augmentation）

代码中对应 `--augmentation appearance`。该配置重点随机化颜色、亮度、对比度、灰度、自动对比度、模糊和锐度，同时不使用 RandomResizedCrop 和 RandomErasing，目的是尽量保留完整人脸结构与细粒度 FAS 纹理。它同样不是已有标准算法名。

对 CASIA 和 Replay-Attack 有改善，但 MSU-MFSD 退化。

运行：

```bash
bash scripts/run_week3_appearance.sh
```

### 3. MixStyle

MixStyle 是已有正式方法名。本项目在 ResNet18 的 `layer1` 和 `layer2` 后插入 MixStyle，以 `p=0.5` 的概率混合 mini-batch 内样本的通道均值和标准差，混合系数服从 `Beta(0.1, 0.1)`；验证和测试阶段自动关闭。

CASIA、Replay-Attack 有改善，但 MSU-MFSD 的 AUC/EER 未优于 Baseline。

运行：

```bash
bash scripts/run_week3_mixstyle.sh
```

### 4. Fourier Amplitude Augmentation（项目轻量实现）

训练阶段对同类别样本进行低频幅度谱混合，并保留原样本相位和大部分高频幅度。默认 `p=0.5`、最大混合权重 `0.35`、低频区域比例 `0.10`。当前实现是本项目基于 Fourier 幅度扰动思想设计的轻量训练增强，不等同于完整复现某篇频域 FAS 方法。

运行：

```bash
bash scripts/run_week3_fourier.sh
```

也可连续运行 MixStyle 和 Fourier：

```bash
bash scripts/run_week3_extra_methods.sh
```

更完整的实现细节见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

### 第三周宏平均结果

| 方法 | 平均 Accuracy | 平均 AUC | 平均 EER | 三域完整改善数 |
|---|---:|---:|---:|---:|
| Baseline | 45.7576% | 42.2235% | 55.3802% | - |
| Strong Data Augmentation | **59.5185%** | 46.9866% | 50.7384% | 1/3 |
| Appearance Randomization | 50.5986% | 47.1715% | 51.7376% | 2/3 |
| MixStyle | 53.5927% | 46.5625% | 52.2578% | 2/3 |
| Fourier Amplitude Augmentation | 58.4002% | **53.7558%** | **48.4219%** | **3/3** |

Fourier 是目前唯一在 CASIA、MSU-MFSD、Replay-Attack 三个目标域上均同时实现 Accuracy 上升、AUC 上升、EER 下降的方法，因此被选为当前推荐方案。

详细第三周报告：[`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)。

## 九、第四周：统一对比、消融式分析与可视化

第四周不再训练新方法，而是统一分析五组实验：

```text
Baseline
Strong Data Augmentation（项目自定义）
Appearance Randomization（项目自定义）
MixStyle
Fourier Amplitude Augmentation（项目轻量实现）
```

运行：

```bash
bash scripts/run_week4_analysis.sh
```

输出目录：[`outputs_week4/`](outputs_week4/)

主要产物：

```text
outputs_week4/
├── week4_all_results.csv
├── week4_macro_summary.csv
├── week4_domain_best.csv
├── week4_analysis.md
├── accuracy_comparison.png
├── auc_comparison.png
├── eer_comparison.png
├── macro_accuracy.png
├── macro_auc.png
├── macro_eer.png
└── improvement_heatmap.png
```

第四周核心结论：

- Strong Data Augmentation 的三域平均 Accuracy 最高；
- Fourier 的三域平均 AUC 最高；
- Fourier 的三域平均 EER 最低；
- Fourier 是唯一三域都实现 Accuracy↑、AUC↑、EER↓ 的方法；
- 因此 Fourier 的跨目标域改善一致性最好。

同时保留限制：CASIA 上 Fourier 的绝对 AUC 仍低于 0.5，因此结论应表述为“稳定改善跨域泛化”，而不是“已经解决跨域泛化”。

详细第四周报告：[`WEEK4_REPORT.md`](WEEK4_REPORT.md)。

## 十、项目结构

```text
FAS/
├── README.md
├── METHOD_IMPLEMENTATION.md
├── EXPERIMENT_RESULTS.md
├── WEEK3_REPORT.md
├── WEEK3_FINAL_REPORT.md
├── WEEK4_REPORT.md
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
│   ├── run_week2_cross_dataset.sh
│   ├── run_week3_generalization.sh
│   ├── run_week3_appearance.sh
│   ├── run_week3_mixstyle.sh
│   ├── run_week3_fourier.sh
│   ├── run_week3_extra_methods.sh
│   └── run_week4_analysis.sh
├── outputs_week4/
├── requirements.txt
└── .gitignore
```

## 十一、实验产物与 GitHub 管理

推荐提交到 GitHub：

- `*.json`：指标、训练历史、数据划分摘要；
- `*.csv`：实验结果汇总；
- `*.md`：阶段报告、自动分析；
- `*.png`：ROC、混淆矩阵、对比图和热力图；
- 实验与分析脚本。

不提交：

- `ProcessedData/`、`RawData/`；
- `*.pth`、`*.pt`、`*.ckpt`；
- 日志、缓存、临时文件。

`.gitignore` 已排除数据集与模型权重，因此可以直接 `git add` 各实验输出目录，权重文件不会被加入。

## 十二、运行环境

当前服务器已验证：

| 参数 | 数值 |
|---|---|
| Python | `3.10` |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0` |
| CUDA | `12.8` |
| GPU | `NVIDIA GeForce RTX 4090 D` |

## 十三、实验记录入口

方法命名与实现细节：[`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)

完整实验结果总表：[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)

第三周最终报告：[`WEEK3_FINAL_REPORT.md`](WEEK3_FINAL_REPORT.md)

第四周完整报告：[`WEEK4_REPORT.md`](WEEK4_REPORT.md)
