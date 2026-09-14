# FAS：基于 ResNet18 的虚假人脸检测 Baseline

本项目用于完成虚假人脸检测（Face Anti-Spoofing, FAS）课程实践。当前阶段以单张 RGB 人脸图像为输入，使用 ResNet18 建立 Live / Spoof 二分类 Baseline，并重点保证数据划分规范、评价指标完整以及实验过程可复现。

项目第一阶段的目标不是设计复杂网络，而是先建立一个可靠的基础模型，为后续跨数据集泛化、数据增强、消融实验和可视化分析提供统一基线。

## 一、任务定义

虚假人脸检测的目标是判断摄像头输入是真实活体人脸还是呈现攻击样本。

当前项目将其建模为二分类问题：

- `Live = 1`：真实活体人脸；
- `Spoof = 0`：打印照片、屏幕回放等虚假人脸攻击。

模型输入为经过人脸裁切后的 RGB 图像，输出两个类别的预测结果。评价阶段使用 Live 类概率计算 ROC、AUC 和 EER。

## 二、当前数据集

默认数据根目录：

```text
/root/Desktop/code/FAS/ProcessedData
```

当前支持的数据集包括：

```text
ProcessedData/
├── CASIA/
│   ├── live/
│   └── spoof/
├── MSU-MFSD/
│   ├── live/
│   └── spoof/
├── OULU-NPU/
│   ├── live/
│   └── spoof/
└── Replay-Attack/
    ├── live/
    └── spoof/
```

数据目录通过 `.gitignore` 排除，不会上传到 GitHub。

数据读取程序会递归扫描图像文件，并根据路径中的 `live`、`real`、`genuine`、`positive` 等关键词识别 Live 类，根据 `spoof`、`attack`、`fake`、`negative` 等关键词识别 Spoof 类。

## 三、数据泄漏控制与 Subject-disjoint 划分

本项目特别关注由视频帧高度相似导致的数据泄漏问题。如果将所有视频抽帧后直接随机划分图像，同一视频中的相邻帧可能同时出现在训练集和测试集中，从而使测试结果明显偏高。

因此当前代码不进行帧级随机划分，而是进一步采用更严格的 Subject-disjoint 策略：同一个 Subject 的所有视频和所有帧只能属于 Train、Validation、Test 中的一个集合。

当前主体 ID 提取逻辑如下：

- OULU-NPU：从文件名中提取 Subject 编号，例如 `1_1_01_1_frame00000.jpg` 中的 `01`；
- MSU-MFSD：提取 `clientXXX`；
- Replay-Attack：提取 `clientXXX`；
- CASIA：根据文件名中的训练/测试前缀和人物编号构造主体分组。

默认划分比例为：

| 参数 | 数值 |
|---|---:|
| `train_ratio` | `0.70` |
| `val_ratio` | `0.15` |
| `test_ratio` | `0.15` |

程序中包含集合交集检查，保证：

```text
Train Subjects ∩ Validation Subjects = ∅
Train Subjects ∩ Test Subjects       = ∅
Validation Subjects ∩ Test Subjects  = ∅
```

需要注意：当前使用的是自定义 Subject-disjoint 70/15/15 划分，不应表述为严格遵循各数据集官方 Protocol。

## 四、模型结构

Baseline 使用 torchvision 提供的 ResNet18。

整体流程为：

```text
RGB 人脸图像
    ↓
Resize 到 224×224
    ↓
数据增强与标准化
    ↓
ResNet18
    ↓
全连接分类层
    ↓
2 类输出
    ↓
Live / Spoof
```

默认使用 ImageNet 预训练权重，并将最后的分类层替换为两个输出节点。

## 五、图像预处理与数据增强

训练阶段当前采用：

- Resize；
- RandomHorizontalFlip；
- ColorJitter；
- ToTensor；
- ImageNet Normalize。

验证和测试阶段不使用随机增强，只进行 Resize、ToTensor 和 Normalize。

主要参数如下：

| 参数 | 数值 |
|---|---|
| `image_size` | `224` |
| `pretrained` | `ImageNet` |
| `num_classes` | `2` |

## 六、训练策略

当前训练阶段使用交叉熵损失和 AdamW 优化器。

| 参数 | 数值 |
|---|---|
| `epochs` | `20` |
| `batch_size` | `64` |
| `lr` | `1e-4` |
| `weight_decay` | `1e-4` |
| `patience` | `5` |
| `seed` | `42` |
| `optimizer` | `AdamW` |

训练过程中每个 epoch 统计训练损失、训练 Accuracy、验证损失和验证 Accuracy。程序根据验证集表现保存最佳模型，并使用 Early Stopping。

## 七、评价指标

当前项目统一输出 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix。

### Accuracy

Accuracy 表示固定分类阈值下预测正确的样本比例，当前默认使用 `threshold = 0.5`。

### AUC

AUC 为 ROC 曲线下面积，用于衡量模型在不同分类阈值下区分 Live 与 Spoof 的整体能力。本项目统一使用 Live 类作为正类，并使用 Live 类预测概率计算 AUC。

### EER

EER 为 Equal Error Rate，即错误接受率与错误拒绝率相等附近对应的错误率。程序同时记录 `eer_threshold`。

### ROC Curve

ROC 曲线展示不同阈值下真正率与假正率之间的关系。

### Confusion Matrix

混淆矩阵用于观察 Spoof 和 Live 的正确分类与错误分类数量。

## 八、第一周：同域 Baseline

第一周正式结果采用 OULU-NPU 的 Subject-disjoint 划分：

| 数据集 | 模型 | 划分方式 | Accuracy | AUC | EER |
|---|---|---|---:|---:|---:|
| OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 99.8161% | 99.9999% | 0.0526% |

混淆矩阵：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

该结果说明模型在 OULU-NPU 同域条件下具有很强的判别能力，但这并不能代表模型具有同样强的跨数据集泛化能力。

## 九、第二周：跨数据集泛化实验

第二周的核心任务是固定第一周训练得到的源模型，直接到完全未参与训练的目标数据集上测试。目标数据集不得用于训练、微调或参数更新。

当前第二周默认设置为：

```text
OULU-NPU 训练得到 ResNet18
          ↓
     固定模型参数
          ↓
CASIA / MSU-MFSD / Replay-Attack 直接测试
```

这样可以观察源域模型在摄像设备、光照环境、攻击介质、图像质量和数据分布发生变化后是否仍然有效。

### 单个跨数据集测试

例如 OULU-NPU -> CASIA：

```bash
python evaluate_cross_dataset.py \
    --checkpoint outputs/OULU-NPU/best.pth \
    --source_dataset OULU-NPU \
    --dataset CASIA \
    --data_root /root/Desktop/code/FAS/ProcessedData
```

输出目录为：

```text
outputs_cross/OULU-NPU_to_CASIA/
```

其中包括跨数据集指标、ROC 曲线、混淆矩阵以及结果摘要 JSON。

### 批量完成第二周实验

推荐直接执行：

```bash
bash scripts/run_week2_cross_dataset.sh
```

脚本会自动测试：

```text
OULU-NPU -> CASIA
OULU-NPU -> MSU-MFSD
OULU-NPU -> Replay-Attack
```

并在全部完成后自动生成：

```text
outputs_cross/cross_dataset_summary.csv
outputs_cross/cross_dataset_summary.md
```

`cross_dataset_summary.csv` 适合后续进行统计或导入表格软件；`cross_dataset_summary.md` 可以直接用于实验报告整理。

### 第二周实验应如何分析

重点将第一周同域性能与第二周跨域性能进行对比：

| 场景 | 训练数据集 | 测试数据集 | Accuracy | AUC | EER |
|---|---|---|---:|---:|---:|
| 同域 | OULU-NPU | OULU-NPU | 99.8161% | 99.9999% | 0.0526% |
| 跨域 | OULU-NPU | CASIA | 待运行 | 待运行 | 待运行 |
| 跨域 | OULU-NPU | MSU-MFSD | 待运行 | 待运行 | 待运行 |
| 跨域 | OULU-NPU | Replay-Attack | 待运行 | 待运行 | 待运行 |

如果跨数据集 Accuracy 和 AUC 明显下降、EER 明显升高，说明模型存在明显的域依赖问题。模型可能学习了源数据集中特有的纹理、颜色、摄像设备或攻击媒介等捷径特征，而不是完全稳定的域无关活体线索。这个结论将直接作为第三周泛化增强实验的出发点。

## 十、项目文件说明

```text
FAS/
├── README.md
├── EXPERIMENT_RESULTS.md
├── datasets.py
├── model.py
├── metrics.py
├── train.py
├── evaluate_cross_dataset.py
├── summarize_cross_results.py
├── scripts/
│   └── run_week2_cross_dataset.sh
├── requirements.txt
├── .gitignore
└── ProcessedData/
```

各文件主要作用如下：

- `datasets.py`：数据扫描、标签识别、Subject ID 提取、Subject-disjoint 划分以及图像预处理；
- `model.py`：构建 ResNet18 二分类模型；
- `metrics.py`：计算 Accuracy、AUC、EER、混淆矩阵并保存 ROC 等评价结果；
- `train.py`：完成训练、验证、Early Stopping、最佳模型保存和同域测试；
- `evaluate_cross_dataset.py`：加载源域模型，在目标数据集上直接执行跨数据集测试；
- `summarize_cross_results.py`：自动汇总多个跨数据集实验结果；
- `scripts/run_week2_cross_dataset.sh`：一键执行第二周全部跨数据集实验；
- `EXPERIMENT_RESULTS.md`：持续记录各阶段实验结果。

## 十一、运行环境

建议使用独立 Conda 环境：

```bash
conda activate fas
```

当前服务器已验证环境：

| 参数 | 数值 |
|---|---|
| `python` | `3.10` |
| `torch` | `2.11.0+cu128` |
| `torchvision` | `0.26.0` |
| `CUDA` | `12.8` |
| `GPU` | `NVIDIA GeForce RTX 4090 D` |

## 十二、实验报告建议记录内容

每次实验至少记录：训练数据集、测试数据集、数据划分方式、模型结构、输入尺寸、主要训练参数、Accuracy、AUC、EER、Confusion Matrix、ROC Curve、最佳 epoch、同域与跨域性能差异以及对性能下降原因的分析。

详细实验结果持续记录在 [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)。
