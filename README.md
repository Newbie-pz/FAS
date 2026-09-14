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
- MSU-MFSD：提取 `clientXXX`，例如 `real_client026_android_SD_scene01_frame00190.jpg` 中的 `client026`；
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

ResNet18 使用残差连接缓解深层网络优化困难，相比更大型网络参数量和计算量较低，适合作为本项目统一 Baseline。

默认使用 ImageNet 预训练权重，然后将最后的分类层替换为两个输出节点。

## 五、图像预处理与数据增强

训练阶段当前采用：

- Resize；
- RandomHorizontalFlip；
- ColorJitter；
- ToTensor；
- ImageNet Normalize。

验证和测试阶段不使用随机增强，只进行 Resize、ToTensor 和 Normalize，从而保证评价过程确定且可复现。

主要参数如下：

| 参数 | 数值 |
|---|---|
| `image_size` | `224` |
| `pretrained` | `ImageNet` |
| `num_classes` | `2` |

## 六、训练策略

当前训练阶段使用交叉熵损失进行二分类优化，并使用 AdamW 优化器。

默认参数如下：

| 参数 | 数值 |
|---|---|
| `epochs` | `20` |
| `batch_size` | `64` |
| `lr` | `1e-4` |
| `weight_decay` | `1e-4` |
| `patience` | `5` |
| `seed` | `42` |
| `optimizer` | `AdamW` |

训练过程中每个 epoch 统计训练损失、训练 Accuracy、验证损失和验证 Accuracy。程序根据验证集表现保存最佳模型，并使用 Early Stopping 避免无意义的持续训练。

## 七、评价指标

当前项目输出 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix。

### Accuracy

Accuracy 表示固定分类阈值下预测正确的样本比例。当前默认使用 `threshold = 0.5`。

### AUC

AUC 为 ROC 曲线下面积，用于衡量模型在不同分类阈值下区分 Live 与 Spoof 的整体能力。AUC 越接近 1，说明两类样本的预测分数整体可分性越强。

本项目统一使用 Live 类作为正类，并使用 Live 类预测概率计算 AUC。

### EER

EER 为 Equal Error Rate，即错误接受率与错误拒绝率相等附近对应的错误率。EER 越低通常表示模型性能越好。

程序同时记录对应的 `eer_threshold`，用于观察 EER 工作点对应的分类阈值。

### ROC Curve

ROC 曲线展示不同阈值下真正率与假正率之间的关系，用于分析模型整体判别能力。

### Confusion Matrix

混淆矩阵用于分别观察 Spoof 和 Live 的正确分类与错误分类数量，便于判断模型错误主要来自哪一类样本。

## 八、第一周正式 Baseline 结果

当前正式结果采用 OULU-NPU 的 Subject-disjoint 划分，不采用早期 Video-level 划分得到的 100% 结果。

| 数据集 | 模型 | 划分方式 | Accuracy | AUC | EER |
|---|---|---|---:|---:|---:|
| OULU-NPU | ResNet18 | Subject-disjoint 70/15/15 | 99.8161% | 99.9999% | 0.0526% |

测试集混淆矩阵：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

本次共测试 11417 张图像，其中 21 个 Spoof 样本被误判为 Live，其余样本分类正确。

详细实验记录见：[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)。

## 九、结果分析时需要注意的问题

当前结果属于同数据集、同域条件下的测试。虽然 Train、Validation 和 Test 的 Subject 严格分离，但它们仍然来自同一个 OULU-NPU 数据集，因此摄像设备、成像过程、攻击介质、光照环境以及数据预处理方式具有较强一致性。

因此，同域 Accuracy 和 AUC 很高并不代表模型具有同样强的跨数据集泛化能力。模型可能同时学习到真正的活体线索以及特定数据集中的纹理、颜色、设备和攻击介质特征。

后续跨数据集实验需要重点观察：

```text
Dataset A 训练
      ↓
ResNet18
      ↓
Dataset B 测试
```

如果跨数据集 Accuracy 和 AUC 明显下降、EER 明显升高，则可以说明模型存在较强的域依赖问题，这也是后续泛化增强方法需要解决的核心问题。

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
├── requirements.txt
├── .gitignore
└── ProcessedData/
```

各文件主要作用如下：

- `datasets.py`：数据扫描、标签识别、Subject ID 提取、Subject-disjoint 划分以及图像预处理；
- `model.py`：构建 ResNet18 二分类模型；
- `metrics.py`：计算 Accuracy、AUC、EER、混淆矩阵并保存 ROC 等评价结果；
- `train.py`：完成训练、验证、Early Stopping、最佳模型保存和测试；
- `evaluate_cross_dataset.py`：加载训练好的模型并在其他数据集上执行跨数据集测试；
- `EXPERIMENT_RESULTS.md`：持续记录不同阶段的实验设置和实验结果；
- `requirements.txt`：Python 依赖说明；
- `.gitignore`：排除数据集、模型权重、日志和临时文件。

## 十一、运行环境

建议使用独立 Conda 环境：

```bash
conda create -n fas python=3.10 -y
conda activate fas
```

当前服务器已验证可用的 PyTorch 环境为：

| 参数 | 数值 |
|---|---|
| `python` | `3.10` |
| `torch` | `2.11.0+cu128` |
| `torchvision` | `0.26.0` |
| `CUDA` | `12.8` |
| `GPU` | `NVIDIA GeForce RTX 4090 D` |

安装其余依赖：

```bash
pip install -r requirements.txt
```

## 十二、训练命令

OULU-NPU Baseline：

```bash
python train.py \
    --dataset OULU-NPU \
    --data_root /root/Desktop/code/FAS/ProcessedData
```

模型训练完成后，结果默认保存在：

```text
outputs/OULU-NPU/
```

主要输出包括：

- `best.pth`：最佳模型参数；
- `history.json`：训练过程记录；
- `split_summary.json`：数据划分信息；
- `test_metrics.json`：测试指标；
- `test_roc.png`：ROC 曲线；
- `test_confusion_matrix.png`：混淆矩阵。

## 十三、跨数据集测试

例如使用 OULU-NPU 训练得到的模型直接测试 CASIA：

```bash
python evaluate_cross_dataset.py \
    --checkpoint outputs/OULU-NPU/best.pth \
    --dataset CASIA \
    --data_root /root/Desktop/code/FAS/ProcessedData
```

跨数据集评价阶段不允许使用目标测试数据更新模型参数，以保证测试集真正作为未见域使用。

## 十四、实验报告建议记录内容

每次实验建议至少记录以下信息：

1. 训练数据集与测试数据集；
2. 数据划分方式及是否保证 Subject-disjoint；
3. 模型结构；
4. 输入尺寸与主要训练参数；
5. Accuracy、AUC、EER；
6. Confusion Matrix；
7. ROC Curve；
8. 最佳 epoch 和 Early Stopping 情况；
9. 与 Baseline 的性能变化；
10. 对错误样本和跨域性能下降原因的分析。

后续所有实验结果建议统一追加到 `EXPERIMENT_RESULTS.md`，避免实验过程中结果散落在终端日志中。