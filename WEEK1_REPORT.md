# 第一周实验报告：数据审计与 OULU-NPU Baseline

## 一、第一周目标

第一周目标是完成虚假人脸检测基础流程：准备训练输入、避免数据泄漏、使用 ResNet18 建立 Live / Spoof 二分类 Baseline，并输出 Accuracy、AUC、EER、ROC Curve 和 Confusion Matrix。

本项目训练仓库的数据入口为已经完成抽帧和人脸区域预处理的 `ProcessedData`。原始视频到 `ProcessedData` 的历史抽帧 FPS、Face Detector 型号和 bbox 扩展比例没有保留，因此本报告不补写无法验证的历史参数。完整边界见 [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md)。

---

## 二、ProcessedData 正式质量审计

使用：

```bash
bash scripts/run_data_audit.sh
```

对四个数据集执行只读审计。正式结果如下：

| 数据集 | 图像数 | Live | Spoof | 无标签 | 无法读取 | 异常小图 | Group 数 | `frame` 命名比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OULU-NPU | 69,896 | 23,060 | 46,836 | 0 | 0 | 0 | 55 | 100% |
| CASIA | 11,110 | 1,810 | 9,300 | 0 | 0 | 0 | 50 | 100% |
| MSU-MFSD | 4,019 | 1,053 | 2,966 | 0 | 0 | 0 | 35 | 100% |
| Replay-Attack | 19,508 | 7,600 | 11,908 | 0 | 0 | 0 | 50 | 100% |
| **合计** | **104,533** | **33,523** | **71,010** | **0** | **0** | **0** | - | **100%** |

四个数据集中的全部图像均为 `256×256`，没有发现无法读取、无法推断标签、无法转换 RGB 或短边低于 64 像素的异常样本。所有样本文件名均具有 `frame` 命名痕迹，能够确认当前训练输入是抽帧后的图像文件。

需要注意：该结果能够证明**最终进入训练的数据质量**，但不能反推出原始视频当时采用的抽帧 FPS、时间间隔、人脸检测器型号或 bbox 扩展比例。

正式审计产物：

```text
outputs_data_audit/
├── processed_data_audit.json
└── processed_data_audit.md
```

---

## 三、数据划分与泄漏控制

课程要求明确避免“先将视频抽帧，再随机按帧划分 Train/Test”的泄漏问题。本项目采用自定义 Subject-disjoint 划分。

默认比例：

| Split | Ratio |
|---|---:|
| Train | 0.70 |
| Validation | 0.15 |
| Test | 0.15 |

代码保证：

```text
Train Subjects ∩ Validation Subjects = ∅
Train Subjects ∩ Test Subjects       = ∅
Validation Subjects ∩ Test Subjects  = ∅
```

OULU-NPU 中当前 Subject/Group 解析得到 55 个 group，满足严格 Subject-disjoint 划分需要。

该划分是本项目自定义实验协议，不应表述为 OULU-NPU 官方 Protocol。

---

## 四、Baseline 模型与训练设置

骨干网络采用 torchvision ResNet18，加载 ImageNet 预训练权重，将最终全连接层替换为 2 类输出。

| 参数 | 设置 |
|---|---|
| 输入尺寸 | `224×224` |
| 类别 | `Spoof=0`, `Live=1` |
| Optimizer | AdamW |
| Learning Rate | `1e-4` |
| Weight Decay | `1e-4` |
| Batch Size | `64` |
| Max Epochs | `20` |
| Early Stopping Patience | `5` |
| Seed | `42` |
| Pretrained | ImageNet |

Baseline 训练增强：

```text
Resize(224×224)
→ RandomHorizontalFlip
→ ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
→ ToTensor
→ ImageNet Normalize
```

验证和测试阶段只执行 Resize、ToTensor 和 ImageNet Normalize。

---

## 五、第一周正式结果

OULU-NPU Subject-disjoint 同域测试：

| Accuracy | AUC | EER | EER Threshold |
|---:|---:|---:|---:|
| **99.8161%** | **99.9999%** | **0.0526%** | 0.643187 |

混淆矩阵：

| 真实类别 | 预测 Spoof | 预测 Live |
|---|---:|---:|
| Spoof | 7609 | 21 |
| Live | 0 | 3787 |

测试集共 11,417 张图像，其中 21 个 Spoof 被预测为 Live，Live 样本全部预测正确。

需要正确解释：第一周结果说明 ResNet18 在当前 OULU-NPU 同域条件下具有很强判别能力，但不能据此推断模型在不同摄像设备、环境或攻击介质下仍具有相同可靠性。因此第二周必须进行跨数据集测试。

---

## 六、第一周正式产物

```text
outputs/OULU-NPU/
├── history.json
├── split_summary.json
├── test_metrics.json
├── test_roc.png
└── test_confusion_matrix.png
```

其中模型权重 `best.pth` 保留在服务器本地，不上传 GitHub。

---

## 七、课程第一周 8 项任务对应关系

| 课程任务 | 当前状态 | 证据/边界 |
|---|---|---|
| ① 准备并检查原始数据 | 已完成最终训练输入审计 | 104,533 张处理后图像，0 损坏、0 无标签、0 异常小图 |
| ② 视频均匀抽帧 | 当前数据明确为抽帧图像 | `frame` 命名比例 100%；历史 FPS/间隔不可追溯 |
| ③ Face Detection / Face Crop | 当前训练输入为已处理人脸图像 | 历史检测器型号与 bbox 参数不可追溯 |
| ④ 检查失败、错误裁切和异常图像 | 已完成自动质量审计 | 可读性、RGB、尺寸、标签均通过；历史人工裁切检查不可追溯 |
| ⑤ Subject/Video 划分 Train/Val/Test | 已完成 | Subject-disjoint 70/15/15 |
| ⑥ ResNet18 Live/Spoof 训练 | 已完成 | `train.py` 与训练历史 |
| ⑦ Accuracy/AUC/EER/ROC/混淆矩阵 | 已完成 | `outputs/OULU-NPU/` |
| ⑧ 第一周实验记录 | 已完成 | 本报告、README、EXPERIMENT_RESULTS |

因此，从当前可验证证据来看，第一周的**训练输入质量检查、数据划分、模型训练、评价指标与实验记录均已闭环**；原始视频预处理部分则明确记录其可追溯边界，不人为补写无法验证的参数。

---

## 八、第一周结论

第一周完成了处理后数据质量审计、Subject-disjoint 数据划分和 ResNet18 Baseline。正式审计覆盖 104,533 张图像，未发现损坏、无标签或异常小图；OULU-NPU 同域 Baseline 达到 99.8161% Accuracy、99.9999% AUC 和 0.0526% EER。

这一结果建立了可靠的同域基线，同时也提出下一阶段的核心问题：**同域接近满分是否意味着模型真正学到了可泛化的活体线索？** 第二周通过未见数据集直接测试该问题。