# 第三周：三方向追加实验说明

本轮实验围绕“尽可能提高跨数据集 AUC”组织，并统一保持目标域不参与训练或参数更新。

## 方向 1：Multi-source DG + ResNet18

使用三个数据集作为源域训练，剩余一个数据集作为未见目标域测试。该实验用于单独观察“多源训练”本身相对单源训练的收益。

## 方向 2：DINOv2-Reg + Multi-source DG

在同一多源协议下将骨干升级为 DINOv2 ViT-B/14 with Registers，并采用：

- source domain × class 平衡采样；
- 每视频均匀选帧；
- 只微调最后一个 Transformer block；
- Focal Loss；
- Video-level AUC 选择 checkpoint。

该路线用于观察更强预训练视觉表征带来的跨域收益。

## 方向 3A：DINOv2-Reg + SSDG-style

SSDG 的核心思想是 Single-Side Domain Generalization：只对 Live 样本做域对抗混淆，同时使用不对称 triplet 结构。

本项目迁移到 DINOv2-Reg 后保留以下核心：

- 所有 Live 样本的 triplet group 统一设为 0；
- 不同源域 Spoof 样本分别属于不同 group；
- 只对 Live feature 使用 gradient reversal + domain discriminator；
- 总损失 = classification + triplet + live-domain adversarial。

因此报告中应称为 `SSDG-style DINOv2 implementation`，不是原论文网络结构的逐行复现。

## 方向 3B：FAS-TD-SF-inspired

FAS-TD-SF/SGTD 原方法依赖多帧信息、空间梯度和虚拟深度监督，官方实现需要 PRNet 生成 depth map。

当前 ProcessedData 只有 RGB 抽帧，因此本项目实现的是 inspired 版本：

前一帧 / 中心帧 / 后一帧 -> DINOv2 CLS 时序差分；中心帧 -> Sobel 空间梯度 -> 小型 CNN；二者拼接后完成分类。

该版本保留“空间梯度 + 多帧动态”的思想，但不包含原论文的 PRNet depth supervision，因此不能写成完整 FAS-TD-SF 复现。

## 与当前 FAS-Aug + PDA + APL 实验的关系

当前正在运行的 `DINOv2-Reg + FAS-Aug + PDA + APL` 是另一条独立的第三周强方案。建议全部实验完成后统一比较 Video AUC，选择效果最高的方法作为第三周正式主结果。

## 一键运行

默认跑 CASIA、MSU-MFSD、Replay-Attack：

    bash scripts/run_week3_three_routes.sh

后台执行：

    mkdir -p logs
    nohup bash scripts/run_week3_three_routes.sh > logs/week3_three_routes.log 2>&1 &

实时查看：

    tail -f logs/week3_three_routes.log

若先只验证 CASIA：

    nohup env TARGETS="CASIA" bash scripts/run_week3_three_routes.sh > logs/week3_three_routes_casia.log 2>&1 &

## 输出

    outputs_week3_three_routes/
    ├── route1_resnet/
    ├── route2_dinov2_train/
    ├── route2_dinov2_eval/
    ├── route3_ssdg_dino/
    ├── route3_td_sf/
    ├── route3_td_sf_eval/
    ├── week3_three_routes_summary.csv
    └── week3_three_routes_summary.md

当前评价优先级：Video AUC > Frame/Sequence AUC > Video EER。
