# FAS Baseline

ResNet18 baseline for RGB face anti-spoofing, following the week-1 project requirements.

## Data
Default root: `/root/Desktop/code/FAS/ProcessedData`

Expected datasets:
- `CASIA`
- `MSU-MFSD`
- `OULU-NPU`
- `Replay-Attack`

The loader recursively searches images and infers labels from path names containing `live/real/genuine/positive` or `spoof/attack/fake/negative`. Frames are grouped by video name before train/val/test splitting to avoid adjacent-frame leakage.

## Install
```bash
pip install -r requirements.txt
```

## Train baseline
```bash
python train.py --dataset OULU-NPU --data_root /root/Desktop/code/FAS/ProcessedData
```

Outputs are written to `outputs/<dataset>/` and include the best checkpoint, metrics JSON, ROC curve and confusion matrix.

## Cross-dataset evaluation
Train on one dataset, then evaluate the saved checkpoint on another:
```bash
python evaluate_cross_dataset.py \
  --checkpoint outputs/OULU-NPU/best.pth \
  --dataset CASIA \
  --data_root /root/Desktop/code/FAS/ProcessedData
```

## Notes
- Label convention: `0 = spoof`, `1 = live`.
- Splitting is performed at inferred video/group level, never by individual frames.
- If a dataset uses nonstandard folder names, pass `--live_keywords` and `--spoof_keywords` to override the defaults.
