# Week 4 Unified Analysis

## Week 3 frozen result

Final model: **Frozen DINOv2-Reg multi-source DG**.

| Target | Video AUC |
|---|---:|
| CASIA | 81.32% |
| MSU-MFSD | 89.01% |
| Replay-Attack | 82.78% |
| **Average** | **84.37%** |

## Analysis assets

- `01_method_comparison_video_auc.png`: Unified cross-domain method comparison
- `02_method_mean_video_auc.png`: Average Video AUC comparison
- `03_tuning_depth_comparison.png`: DINOv2 tuning-depth diagnostic
- `04_ssdg_parameter_sensitivity.png`: SSDG-style parameter sensitivity
- `05_fusion_mean_video_auc.png`: Fusion strategy comparison
- `06_source_target_generalization_gap.png`: Source-vs-target generalization gap
- `06_final_model_video_roc.png`: Final model video-level ROC curves
- `07_confusion_CASIA.png`: CASIA confusion matrix
- `07_confusion_MSU-MFSD.png`: MSU-MFSD confusion matrix
- `07_confusion_Replay-Attack.png`: Replay-Attack confusion matrix

## Interpretation notes

- The freeze=10/11/12 comparison is a diagnostic tuning-depth comparison rather than a perfectly controlled ablation, because the historical runs used different backbone learning rates.
- The SSDG-style variants are suitable for parameter-sensitivity analysis; the frozen SSDG configuration is especially useful to illustrate target-dependent gains and failure on Replay-Attack.
- Fusion results should be reported as an analysis rather than replacing the frozen Week-3 main result.
- Source validation AUC near 100% together with substantially lower unseen-target AUC directly visualizes the source-domain overfitting/generalization gap.
- Video-level ROC curves are regenerated from the saved final-model video scores, so no retraining is required.
