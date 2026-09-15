# 方法命名与实现细节

本文档用于说明本项目各实验方法的**名称来源、代码实现、关键参数、训练/测试行为和技术报告中的推荐表述**。目的有两个：

1. 避免把项目内部定义的实验配置误写成已有论文中的标准算法名；
2. 为后续技术报告、答辩、复现实验提供可以直接核对代码的实现说明。

> **重要命名约定**：`Strong Data Augmentation` 与 `Appearance Randomization` 都是本项目为了组织实验而定义的训练策略名称，并不是具有唯一标准定义的公开算法名。`MixStyle` 是已有正式方法名；`Fourier Amplitude Augmentation` 在本项目中指一个轻量、项目自定义的频域幅度增强实现，不等同于完整复现某篇频域 FAS 论文的全部方法。

---

## 一、方法总览与命名性质

| 文档名称 | 代码设置 | 名称性质 | 是否可直接当作已有标准算法名 |
|---|---|---|---|
| Baseline | `--augmentation baseline --method baseline` | 本项目基础训练配置 | 否 |
| Strong Data Augmentation | `--augmentation strong --method baseline` | **本项目自定义强增强配置** | 否 |
| Appearance Randomization | `--augmentation appearance --method baseline` | **本项目自定义外观随机化策略** | 否 |
| MixStyle | `--augmentation baseline --method mixstyle` | 已有正式方法名；本项目做 ResNet18 轻量集成 | 是，但技术报告中应说明本项目的具体插入位置与参数 |
| Fourier Amplitude Augmentation | `--augmentation baseline --method fourier` | **本项目自定义的轻量频域增强实现** | 不应写成“完整复现某篇频域 FAS 方法” |

历史文档中曾使用 `Strong Augmentation` 和 `Appearance Augmentation`。为后续技术报告表达更准确，推荐统一写为：

- `Strong Data Augmentation`：强数据增强；
- `Appearance Randomization`：外观随机化。

其中 `Appearance Augmentation` 与 `Appearance Randomization` 指向**同一个代码配置**，只是后者更能准确表达设计意图。

---

## 二、共同实验设置

除特别说明外，各方法保持以下条件一致：

- 骨干网络：ResNet18；
- 输入：RGB 人脸图像，`224 × 224`；
- ImageNet 预训练权重；
- 分类类别：`Spoof = 0`，`Live = 1`；
- 损失函数：Cross Entropy；
- 优化器：AdamW；
- `learning rate = 1e-4`；
- `weight_decay = 1e-4`；
- `batch_size = 64`；
- 最大训练轮数：20；
- Early Stopping patience：5；
- 随机种子：42；
- OULU-NPU 使用自定义 Subject-disjoint `70/15/15` 划分；
- 跨数据集实验采用 Source-only 协议：目标域不参与训练、微调或参数更新。

验证集、同域测试集和跨域目标数据**均不执行随机训练增强**，统一只进行：

```text
Resize(224, 224)
→ ToTensor
→ ImageNet Normalize
```

因此不同方法之间的主要差异发生在训练阶段。

---

## 三、Baseline

### 3.1 方法定位

Baseline 不是某个特殊算法，而是本项目用于建立统一比较基准的 ResNet18 训练配置。

### 3.2 训练增强

代码位置：`datasets.py -> build_train_transform(..., augmentation='baseline')`

实际顺序：

```text
Resize(224, 224)
→ RandomHorizontalFlip()
→ ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
→ ToTensor()
→ Normalize(ImageNet mean/std)
```

其中未显式指定概率的 `RandomHorizontalFlip()` 使用 torchvision 默认概率 `p=0.5`。

### 3.3 推荐技术报告表述

> Baseline 采用 ImageNet 预训练 ResNet18，在训练阶段使用水平翻转和轻量颜色扰动，不引入额外域泛化模块。

---

## 四、Strong Data Augmentation

### 4.1 命名说明

`Strong Data Augmentation` 是**本项目自定义的强数据增强配置名称**。它不是一个具有统一论文定义或固定实现的独立算法。

它的实验目的不是提出新的网络结构，而是验证：通过显著扩大训练图像在空间、颜色、清晰度和局部可见区域上的变化，是否能够降低模型对 OULU-NPU 单一源域统计特征的依赖。

代码参数：

```bash
--augmentation strong --method baseline
```

### 4.2 具体实现

代码位置：`datasets.py -> build_train_transform(..., augmentation='strong')`

实际处理流程：

```text
RandomResizedCrop(
    size=224,
    scale=(0.80, 1.00),
    ratio=(0.90, 1.10)
)
→ RandomHorizontalFlip(p=0.5)
→ RandomApply(
      ColorJitter(
          brightness=0.40,
          contrast=0.40,
          saturation=0.40,
          hue=0.10
      ),
      p=0.80
  )
→ RandomGrayscale(p=0.15)
→ RandomApply(
      GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
      p=0.30
  )
→ ToTensor()
→ Normalize(ImageNet mean/std)
→ RandomErasing(
      p=0.25,
      scale=(0.02, 0.15),
      ratio=(0.3, 3.3),
      value='random'
  )
```

### 4.3 每个算子的作用

- `RandomResizedCrop`：改变有效人脸区域占比、局部尺度和裁切位置；
- `ColorJitter`：扩大亮度、对比度、饱和度和色调变化；
- `RandomGrayscale`：降低模型对固定 RGB 颜色统计的依赖；
- `GaussianBlur`：模拟不同设备、焦距、压缩或成像清晰度；
- `RandomErasing`：随机遮挡局部区域，降低模型对单一区域的过度依赖。

### 4.4 设计风险

FAS 与普通物体分类不同，真假人脸判断可能依赖细粒度纹理、屏幕摩尔纹、打印纹理和局部反射等线索。因此过强的随机裁切、模糊和擦除也可能**破坏真正有用的 spoof cue**。本项目实验中 Strong 在 CASIA 上提升明显，但在 MSU-MFSD 和 Replay-Attack 上并没有形成稳定 AUC/EER 改善，与这一风险相符。

### 4.5 推荐技术报告表述

> Strong Data Augmentation 为本项目定义的强增强训练配置，由随机尺度裁切、较强颜色扰动、灰度化、模糊和随机擦除组成，用于扩大源域样本外观和空间变化。该名称是实验配置名称，而非已有标准算法名。

---

## 五、Appearance Randomization

### 5.1 命名说明

`Appearance Randomization` 是**本项目自定义的外观/成像风格随机化策略**。早期实验记录中称为 `Appearance Augmentation`，两者对应完全相同的代码配置。

之所以推荐改称 `Appearance Randomization`，是因为该配置的核心目标不是简单“增加增强强度”，而是**尽量保持完整人脸结构，同时随机化颜色、对比度、灰度、模糊与锐度等成像外观因素**。

代码参数：

```bash
--augmentation appearance --method baseline
```

### 5.2 具体实现

代码位置：`datasets.py -> build_train_transform(..., augmentation='appearance')`

实际处理流程：

```text
Resize(224, 224)
→ RandomHorizontalFlip(p=0.5)
→ RandomApply(
      ColorJitter(
          brightness=0.35,
          contrast=0.35,
          saturation=0.30,
          hue=0.05
      ),
      p=0.80
  )
→ RandomGrayscale(p=0.10)
→ RandomAutocontrast(p=0.15)
→ RandomApply(
      GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
      p=0.15
  )
→ RandomAdjustSharpness(
      sharpness_factor=0.6,
      p=0.10
  )
→ ToTensor()
→ Normalize(ImageNet mean/std)
```

### 5.3 与 Strong 的关键区别

Appearance Randomization **不使用**：

```text
RandomResizedCrop
RandomErasing
```

因此它不会主动改变人脸区域的整体空间构成，也不会随机抹去局部区域。相较 Strong，它是一套更克制的增强策略，重点扰动：

- 亮度；
- 对比度；
- 饱和度；
- 色调；
- 灰度分布；
- 自动对比度；
- 模糊程度；
- 图像锐度。

设计动机是：FAS 的局部纹理可能具有判别价值，因此尽量不破坏空间细节，而重点削弱设备、光照和成像风格相关的外观偏置。

### 5.4 推荐技术报告表述

> Appearance Randomization 为本项目定义的外观随机化训练策略，主要扰动颜色、对比度、灰度、模糊和锐度，同时避免随机裁切和局部擦除，以尽量保留 FAS 所需的细粒度人脸纹理。该名称为项目实验配置名称，并非已有标准算法名。

---

## 六、MixStyle

### 6.1 命名说明

`MixStyle` 是已有正式方法名。本项目没有重新命名该方法，而是在 ResNet18 中做了一个轻量集成。

需要注意：技术报告中即使使用正式名称 `MixStyle`，也仍应说明本项目的**插入位置、概率和 Beta 分布参数**，不要只写“使用 MixStyle”。

代码参数：

```bash
--augmentation baseline \
--method mixstyle \
--mixstyle_p 0.5 \
--mixstyle_alpha 0.1
```

### 6.2 插入位置

代码位置：`model.py`

当前 ResNet18 前向传播：

```text
conv1 → bn1 → relu → maxpool
→ layer1
→ MixStyle #1
→ layer2
→ MixStyle #2
→ layer3
→ layer4
→ avgpool → fc
```

因此 MixStyle 只作用于较浅层特征，`layer3` 和 `layer4` 后不再执行风格统计混合。

### 6.3 计算过程

对于批次特征 `x ∈ R^(B×C×H×W)`，首先计算每个样本、每个通道的空间均值和标准差：

```text
μ = mean(x, H, W)
σ = sqrt(var(x, H, W) + eps)
```

标准化：

```text
x_norm = (x - μ) / σ
```

随机打乱批次得到 donor 样本，并从 Beta 分布采样：

```text
λ ~ Beta(alpha, alpha)
alpha = 0.1
```

混合风格统计：

```text
μ_mix = λ μ + (1 - λ) μ_donor
σ_mix = λ σ + (1 - λ) σ_donor
```

最后恢复：

```text
x_out = x_norm * σ_mix + μ_mix
```

当前激活概率：

```text
p = 0.5
```

即训练时每次经过 MixStyle 层，以 50% 概率执行统计混合。

### 6.4 训练与测试行为

`MixStyle.forward()` 中包含：

```text
if not self.training: return x
```

因此：

- 训练阶段：随机执行 MixStyle；
- 验证、同域测试、跨域测试：MixStyle 自动关闭，不修改特征。

当前 donor 由当前 mini-batch 中随机排列得到，没有使用目标域信息。

### 6.5 推荐技术报告表述

> 在 ResNet18 的 `layer1` 和 `layer2` 后插入 MixStyle 模块，以 `p=0.5` 的概率混合 mini-batch 内样本的通道均值和标准差，混合系数服从 `Beta(0.1, 0.1)`。MixStyle 仅在训练阶段启用，测试阶段关闭。

---

## 七、Fourier Amplitude Augmentation

### 7.1 命名说明

本项目中的 `Fourier Amplitude Augmentation` 是一个**轻量、项目自定义的频域训练增强实现**。它使用 Fourier 幅度/相位分解和低频幅度混合思想，但并不是完整复现某篇频域 FAS 论文中的全部网络结构、蒸馏策略或损失函数。

因此技术报告中推荐写：

> “本项目实现了一种 lightweight Fourier amplitude augmentation”

而不要写：

> “完整复现了某某频域 FAS 方法”。

代码参数：

```bash
--augmentation baseline \
--method fourier \
--fourier_p 0.5 \
--fourier_max_lambda 0.35 \
--fourier_low_freq_ratio 0.10
```

### 7.2 训练阶段处理流程

代码位置：`dg_methods.py -> fourier_amplitude_mix()` 与 `train.py -> run_epoch()`。

该操作只在：

```text
train == True and method == 'fourier'
```

时执行。

完整流程：

```text
1. 输入首先经过 Baseline 图像增强并完成 ImageNet Normalize
2. 反归一化回 [0, 1] RGB 图像
3. 对 H、W 两个空间维度执行 fft2
4. 分解为 amplitude（幅度）与 phase（相位）
5. fftshift，将低频移动到频谱中心
6. 在当前 mini-batch 内按同类别构建 donor permutation
7. 仅对频谱中心的低频 amplitude patch 做线性混合
8. 保留原样本 phase 和未被修改的高频 amplitude
9. ifftshift
10. 用 mixed amplitude + original phase 重建复频谱
11. ifft2 回到图像空间
12. clamp 到 [0, 1]
13. 再执行 ImageNet Normalize
```

### 7.3 同类别 donor 约束

本项目没有在 Live 与 Spoof 之间随机互换幅度谱，而是：

```text
Live  ↔ Live
Spoof ↔ Spoof
```

对应函数：

```text
_same_class_permutation(labels)
```

如果一个类别在当前 mini-batch 中只有一个样本，该样本会保留自身作为 donor。

设计原因：FAS 的频域特征本身可能包含攻击介质纹理。如果直接进行 Live ↔ Spoof 的跨类别幅度混合，可能把与标签有关的 spoof cue 混入另一类别，导致标签语义污染。因此当前实现只把幅度混合作为**类内域风格随机化**。

### 7.4 低频区域与混合系数

默认参数：

```text
fourier_p = 0.5
fourier_max_lambda = 0.35
fourier_low_freq_ratio = 0.10
```

含义：

- 每个训练 batch 以 50% 概率执行 Fourier augmentation；
- 每个样本的混合系数 `λ` 从 `[0, 0.35)` 均匀采样；
- 只修改频谱中心约 10% 尺度的低频区域。

低频幅度混合公式：

```text
A_mix = (1 - λ) * A_source + λ * A_donor
```

相位保持：

```text
P_mix = P_source
```

重建：

```text
X_mix = IFFT(A_mix, P_source)
```

### 7.5 设计动机与边界

本项目的工作假设是：部分跨数据集差异与整体颜色、亮度、成像设备和低频外观统计有关，因此只扰动低频幅度可以扩大源域风格分布；同时保留原始相位与大部分高频幅度，尽量减少对人脸结构和局部 spoof 纹理的破坏。

需要避免过度解释：当前结果支持“该实现能够稳定改善本实验设置下的三域泛化”，但不能仅凭本实验证明“低频幅度就是所有 FAS 域偏移的根本原因”。

### 7.6 推荐技术报告表述

> 在训练阶段对 ImageNet 归一化后的输入进行反归一化和二维 FFT，将图像分解为幅度与相位。以 `p=0.5` 的概率，在同类别 mini-batch 样本之间混合频谱中心约 10% 范围内的低频幅度，混合权重最大为 0.35，同时保留源样本相位和其余高频幅度。重建后重新归一化并送入 ResNet18。验证和测试阶段不使用该增强。

---

## 八、五种方法的变量控制关系

| 方法 | 图像空间增强 | 特征空间改动 | 频域改动 | 测试阶段额外处理 |
|---|---|---|---|---|
| Baseline | 基础 | 无 | 无 | 无 |
| Strong Data Augmentation | 强 | 无 | 无 | 无 |
| Appearance Randomization | 外观随机化 | 无 | 无 | 无 |
| MixStyle | Baseline | `layer1/layer2` 后统计混合 | 无 | 无，MixStyle 自动关闭 |
| Fourier Amplitude Augmentation | Baseline | 无 | 训练期低频幅度类内混合 | 无 |

这一设计使不同实验可以从“训练阶段增加了什么机制”的角度进行对照。

---

## 九、复现实验命令

### Baseline

```bash
python train.py \
  --dataset OULU-NPU \
  --augmentation baseline \
  --method baseline
```

### Strong Data Augmentation

```bash
python train.py \
  --dataset OULU-NPU \
  --augmentation strong \
  --method baseline
```

### Appearance Randomization

```bash
python train.py \
  --dataset OULU-NPU \
  --augmentation appearance \
  --method baseline
```

### MixStyle

```bash
python train.py \
  --dataset OULU-NPU \
  --augmentation baseline \
  --method mixstyle \
  --mixstyle_p 0.5 \
  --mixstyle_alpha 0.1
```

### Fourier Amplitude Augmentation

```bash
python train.py \
  --dataset OULU-NPU \
  --augmentation baseline \
  --method fourier \
  --fourier_p 0.5 \
  --fourier_max_lambda 0.35 \
  --fourier_low_freq_ratio 0.10
```

项目已有一键脚本位于 `scripts/`，正式实验优先使用这些脚本，以避免输出目录和参数不一致。

---

## 十、技术报告中的推荐命名

后续技术报告、课程答辩或论文式总结中，推荐第一次出现时写全称并说明性质：

- **Baseline ResNet18**：项目基础配置；
- **Strong Data Augmentation (project-defined)**：项目自定义强数据增强配置；
- **Appearance Randomization (project-defined)**：项目自定义外观随机化策略；
- **MixStyle**：已有特征统计域泛化方法，本项目插入 ResNet18 `layer1/layer2` 后；
- **Lightweight Fourier Amplitude Augmentation (project implementation)**：项目实现的轻量频域幅度增强。

后文可简写为：

```text
Baseline / Strong / Appearance / MixStyle / Fourier
```

但表格或正文第一次出现时应避免只写 `Strong`、`Appearance`，否则容易被误解为具有固定公开定义的方法名。

---

## 十一、代码对应关系

| 内容 | 代码文件 |
|---|---|
| Baseline / Strong / Appearance 图像增强 | `datasets.py` |
| MixStyle 计算 | `dg_methods.py` |
| MixStyle 插入 ResNet18 | `model.py` |
| Fourier amplitude mix | `dg_methods.py` |
| Fourier 训练期调用逻辑 | `train.py` |
| 跨域测试 | `evaluate_cross_dataset.py` |
| 第四周统一分析 | `analyze_week4.py` |

如果后续修改任何增强参数，应同时更新本文档和对应实验报告，避免“代码实现”与“技术报告描述”不一致。
