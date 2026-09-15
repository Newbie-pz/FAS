# 数据预处理、质量检查与数据来源说明

本文档用于说明本项目从原始视频/图像到训练输入之间的数据处理边界，并明确哪些步骤能够由当前仓库直接复现、哪些步骤属于实验开始前已经完成的预处理。该说明主要用于补齐课程第一周“数据准备—抽帧—人脸裁切—异常检查—数据划分”的技术记录，避免后续技术报告中把没有证据支持的预处理细节写成既成事实。

## 一、当前仓库的数据入口

当前训练与评估代码不直接读取原始视频，而是从以下目录开始工作：

```text
/root/Desktop/code/FAS/ProcessedData
```

目录结构为：

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

这些目录中的样本已经是图像帧，文件名保留了原视频/Subject/Client 等信息，例如：

```text
OULU-NPU/live/1_1_01_1_frame00000.jpg
MSU-MFSD/live/real_client026_android_SD_scene01_frame00190.jpg
Replay-Attack/live/client030_session01_webcam_authenticate_adverse_1_frame00020.jpg
CASIA/live/test_26_1_frame00110.jpg
```

因此，本项目当前可直接证明：模型训练输入来自已经抽帧并整理为 Live/Spoof 两类的处理后图像。

## 二、关于原始视频抽帧与 Face Crop 的可追溯性

课程第一周要求包含：

1. 准备并检查原始数据；
2. 视频均匀抽帧；
3. 使用人脸检测器进行 Face Crop；
4. 检查检测失败、错误裁切和异常图像；
5. 按 Subject/Video 划分 Train/Validation/Test；
6. 使用 ResNet18 完成二分类；
7. 输出 Accuracy、AUC、EER、ROC、Confusion Matrix；
8. 提交实验记录。

当前仓库能够完整复现第 5～8 项，并已经对第 1、4 项完成事后数据质量审计；但**原始视频到 `ProcessedData` 的抽帧与人脸裁切流程并未保留在当前仓库中**。

因此，当前技术报告中不应虚构以下信息：

- 不应声称使用了某个具体的人脸检测器，除非另有原始记录；
- 不应声称使用了某个固定抽帧间隔或 FPS，除非另有原始记录；
- 不应声称 bbox 按某个比例扩展，除非另有原始记录；
- 不应把当前训练仓库描述成“从原始视频端到端完成全部预处理”。

推荐在技术报告中使用如下表述：

> 本项目训练与评估阶段使用已经完成视频抽帧与人脸区域预处理的 `ProcessedData` 作为数据入口。当前仓库重点负责标签读取、Subject ID 解析、Subject-disjoint 数据划分、模型训练、跨数据集评估以及域泛化实验。原始视频到处理后图像的抽帧和 Face Crop 参数未纳入当前仓库，因此不对未保留记录的检测器型号、抽帧频率或检测框扩展参数作推断性描述。

## 三、当前仓库的数据质量审计

脚本：

```text
scripts/audit_processed_data.py
```

检查内容包括：

1. 文件可读性；
2. RGB 可转换性；
3. Live/Spoof 标签识别；
4. 类别数量；
5. Subject/Group 数；
6. 图像尺寸与异常小图；
7. 文件名中的 `frame` 抽帧痕迹；
8. Subject-disjoint 可划分性。

脚本不会修改任何图像，只做只读审计。

运行：

```bash
cd /root/Desktop/code/FAS
conda activate fas
bash scripts/run_data_audit.sh
```

正式结果保存于：

```text
outputs_data_audit/
├── processed_data_audit.json
└── processed_data_audit.md
```

## 四、正式审计结果

本项目已经对实际 `ProcessedData` 完成正式审计，覆盖 **104,533 张图像**。

| 数据集 | 图像数 | Live | Spoof | 无标签 | 无法读取 | 异常小图 | Group 数 | `frame` 命名比例 | 可进行 Subject-disjoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OULU-NPU | 69,896 | 23,060 | 46,836 | 0 | 0 | 0 | 55 | 100% | 是 |
| CASIA | 11,110 | 1,810 | 9,300 | 0 | 0 | 0 | 50 | 100% | 是 |
| MSU-MFSD | 4,019 | 1,053 | 2,966 | 0 | 0 | 0 | 35 | 100% | 是 |
| Replay-Attack | 19,508 | 7,600 | 11,908 | 0 | 0 | 0 | 50 | 100% | 是 |
| **合计** | **104,533** | **33,523** | **71,010** | **0** | **0** | **0** | - | **100%** | - |

四个数据集的图像分辨率全部为：

```text
256 × 256
```

正式审计中没有发现：

- 无法读取图像；
- 无法推断 Live/Spoof 标签的图像；
- 无法转换 RGB 的图像；
- 短边小于 64 像素的异常小图。

因此，从当前训练输入层面，可以认为数据文件的**可读性、标签结构和尺寸一致性均通过检查**。

同时，四个数据集的 `frame` 文件名比例均为 100%，这能够证明当前目录中的样本是抽帧后的图像；但仍不能据此反推出历史抽帧 FPS 或时间间隔。

## 五、Subject ID 与数据泄漏控制

当前项目最重要的可复现数据协议不是“帧级随机划分”，而是 **Subject-disjoint**。

`datasets.py` 会根据不同数据集的文件名规则解析 Subject/Client：

- OULU-NPU：从 `1_1_01_1_frame00000.jpg` 中提取 `01`；
- MSU-MFSD：提取 `clientXXX`；
- Replay-Attack：提取 `clientXXX`；
- CASIA：根据 `train/test` 前缀和人物编号构造 group。

正式审计识别到的 Group 数为：

| 数据集 | Group 数 |
|---|---:|
| OULU-NPU | 55 |
| CASIA | 50 |
| MSU-MFSD | 35 |
| Replay-Attack | 50 |

各数据集均具备至少 3 个独立 Group，因此满足严格 Train/Validation/Test 分离的基本条件。

训练代码保证：

```text
Train Subjects ∩ Validation Subjects = ∅
Train Subjects ∩ Test Subjects       = ∅
Validation Subjects ∩ Test Subjects  = ∅
```

默认划分比例：

| Split | Ratio |
|---|---:|
| Train | 0.70 |
| Validation | 0.15 |
| Test | 0.15 |

需要注意：该协议是本项目的自定义 Subject-disjoint 70/15/15 划分，不能表述为各数据集官方 Protocol。

## 六、训练阶段的图像预处理

无论第三周采用哪一种泛化策略，验证/测试阶段都使用固定预处理：

```text
Resize(image_size × image_size)
→ ToTensor
→ ImageNet Normalize
```

默认：

```text
image_size = 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Baseline 训练阶段使用：

```text
Resize(224×224)
→ RandomHorizontalFlip
→ ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
→ ToTensor
→ ImageNet Normalize
```

第三周其它训练增强的完整参数见 [`METHOD_IMPLEMENTATION.md`](METHOD_IMPLEMENTATION.md)。

## 七、第一周任务完成度的严谨表述

| 课程任务 | 当前证据状态 |
|---|---|
| 准备并检查原始数据 | 已有 `ProcessedData`；已完成 104,533 张最终训练输入的正式质量审计 |
| 视频均匀抽帧 | `frame` 命名比例 100%，确认当前样本为抽帧图像；历史 FPS/间隔未保留 |
| Face Detection / Face Crop | 当前训练输入为已处理图像；历史检测器与 bbox 参数未保留 |
| 检测失败/错误裁切/异常检查 | 正式审计确认 0 损坏、0 无标签、0 异常小图、0 RGB 转换失败；历史人工裁切检查不可追溯 |
| Subject/Video 划分 | 已实现 Subject-disjoint 70/15/15 |
| ResNet18 训练 | 已完成 |
| Accuracy/AUC/EER/ROC/Confusion Matrix | 已完成 |
| 第一周实验记录 | 已完成并上传 GitHub |

因此，项目在**最终训练输入质量、数据划分、模型训练、评价和后续跨域实验**方面已经闭环；原始视频预处理部分则采用“明确数据入口 + 诚实记录不可追溯项 + 当前正式质量审计”的方式进行技术说明。

第一周独立报告见 [`WEEK1_REPORT.md`](WEEK1_REPORT.md)。

## 八、答辩中的推荐表述

如果老师询问“抽帧和人脸裁切是怎么做的”，推荐回答：

> 当前实验仓库从已经完成抽帧和人脸区域预处理的 `ProcessedData` 开始，后续 Subject-disjoint 划分、训练、跨域测试和泛化实验均可完整复现。原始视频到处理后图像的具体检测器型号和抽帧参数没有纳入当前代码仓库，因此报告中没有人为补写无法验证的参数。我们额外对实际训练输入进行了完整数据质量审计，共检查 104,533 张图像，未发现损坏、无标签或异常尺寸样本，四个数据集均具备可用于 Subject-disjoint 划分的独立 Group。

这一表述能够准确说明项目边界，同时保持技术报告的可验证性。