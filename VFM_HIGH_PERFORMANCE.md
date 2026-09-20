# 高性能跨域 FAS 路线：DINOv2 with Registers + Multi-source DG

本文档记录用于提升跨数据集 AUC 的新实验路线。旧版 ResNet18 单源实验不删除，继续作为“极端单源压力测试”；本路线单独作为高性能多源域泛化实验。

## 1. 为什么新增这条路线

旧协议为：

```text
OULU-NPU -> CASIA
OULU-NPU -> MSU-MFSD
OULU-NPU -> Replay-Attack
```

它只有一个源域，跨域 AUC 明显偏低。新路线采用 MICO 风格的 leave-one-domain-out 多源域泛化：

```text
OULU + MSU + Replay -> CASIA
OULU + CASIA + Replay -> MSU
OULU + CASIA + MSU -> Replay
```

目标域严格不进入训练、验证或参数选择。

## 2. 核心模型

骨干：

```text
DINOv2 ViT-B/14 with Registers
torch.hub model: dinov2_vitb14_reg
```

本项目分类特征默认使用：

```text
CLS token
+
所有 patch token 的均值
-> concatenate
-> LayerNorm
-> Dropout
-> Linear(2)
```

这样既保留全局语义，也显式加入局部 patch 表征。

## 3. 为效果做的训练协议调整

### 3.1 Domain × Class balanced sampling

四个数据集规模差异很大。若直接混合训练，OULU-NPU 会在 batch 中占据绝大多数样本。

因此训练 sampler 对每个：

```text
(source domain, Live/Spoof)
```

组合赋予相同总采样质量，避免大数据源主导梯度。

### 3.2 每视频均匀抽取训练帧

ProcessedData 中连续视频帧高度相关。新训练流程默认每个源视频最多均匀抽取 20 帧：

```text
max_train_frames_per_video = 20
max_val_frames_per_video   = 20
```

这样可以降低相邻帧重复、降低长视频权重、加快 ViT 微调，并减少模型记忆源视频背景和成像模式。

### 3.3 Source validation 保持 Subject-disjoint

每个源域内部按照 Subject/Client 划分训练/验证：

```text
source train : 90%
source val   : 10%
```

同一个 Subject 不会同时进入 source train 与 source validation。

### 3.4 最佳 checkpoint 由 Video-level AUC 决定

不再根据 frame accuracy 或 validation loss 选择最佳模型。

优先级：

```text
更高 Video AUC
若相同 -> 更低 Video EER
```

## 4. 默认训练设置

| 参数 | 数值 |
|---|---:|
| Backbone | DINOv2 ViT-B/14 Registers |
| image size | 224 |
| epochs | 15 |
| batch size | 24 |
| gradient accumulation | 2 |
| effective batch | 48 |
| backbone LR | 1e-5 |
| head LR | 2e-4 |
| weight decay | 0.05 |
| label smoothing | 0.05 |
| warmup | 5% total steps |
| AMP | enabled |
| max frames/video | 20 |
| best model metric | source-val Video AUC |

骨干默认全量微调。若显存不足，可以通过 `--freeze_first_blocks` 冻结前若干 Transformer blocks，但效果优先时首先尝试全量微调。

## 5. 评价方式

每个目标域同时输出 Frame-level 和 Video-level 两套 Accuracy/AUC/EER。

Video-level 根据文件名恢复原视频 ID：

```text
*_frameXXXXX.jpg -> original video id
```

对同一视频的 Live 概率取均值，再计算 Video AUC / EER。后续高性能路线以 Video-level 指标为主，Frame-level 作为补充。

## 6. 代码文件

```text
model_vfm.py
vfm_data.py
train_vfm_multisource.py
evaluate_vfm.py
summarize_vfm_results.py
scripts/run_week2_vfm_mico.sh
```

旧代码不删除。

## 7. 首次建议：先跑 CASIA

CASIA 是当前旧实验最困难的目标域，因此先把它作为 canary：

```bash
cd /root/Desktop/code/FAS
git pull
conda activate fas

TARGETS="CASIA" GPU=0 bash scripts/run_week2_vfm_mico.sh
```

重点看最终输出中的 `VIDEO AUC` 和 `VIDEO EER`。

如果 CASIA 已经出现大幅跃升，再完整运行三个目标域：

```bash
TARGETS="CASIA MSU-MFSD Replay-Attack" GPU=0 bash scripts/run_week2_vfm_mico.sh
```

## 8. 显存不足时

首先降低 batch，而不是降低模型：

```bash
BATCH_SIZE=12 GRAD_ACCUM=4 TARGETS="CASIA" GPU=0 bash scripts/run_week2_vfm_mico.sh
```

有效 batch 仍保持约 48。

## 9. 结果边界

旧版单源 ResNet18 与新多源 DINOv2-Reg 不属于同一训练协议，因此不能写成“只替换 backbone 的公平消融”。

推荐表述：

- 旧实验：single-source stress test；
- 新实验：high-performance multi-source DG / MICO-style protocol。

如果新路线达到预期，再以该 DINOv2-Reg checkpoint 为基础开展第三周 FAS-specific augmentation / local patch supervision，而不是继续围绕旧 ResNet18 调参。
