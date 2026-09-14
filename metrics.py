import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float('nan')
    eer, eer_threshold = compute_eer(y_true, y_score) if len(np.unique(y_true)) == 2 else (float('nan'), float('nan'))
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'auc': float(auc),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'threshold': float(threshold),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_plots_and_metrics(y_true, y_score, out_dir, prefix='test', threshold=0.5):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_score, threshold)
    with open(out / f'{prefix}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f'{prefix}_roc.png', dpi=200)
        plt.close()

    cm = np.array(metrics['confusion_matrix'])
    plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ['Spoof', 'Live'])
    plt.yticks([0, 1], ['Spoof', 'Live'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out / f'{prefix}_confusion_matrix.png', dpi=200)
    plt.close()
    return metrics
