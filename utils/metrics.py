"""完整的评价指标：OA、mIoU、Kappa、F1、混淆矩阵。"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> dict:
    """计算完整的分类指标。

    Args:
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        num_classes: 类别数

    Returns:
        dict 包含: OA, mIoU, kappa, f1_crop, iou_crop, conf_matrix
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # OA
    oa = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    # 每类 IoU
    ious = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    ious = np.array(ious)
    miou = np.nanmean(ious)

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # F1 (耕地类, class=1)
    f1_crop = f1_score(y_true, y_pred, labels=[1], average=None)[0] if 1 in y_true else 0.0

    # IoU (耕地类)
    iou_crop = ious[1] if num_classes > 1 else float("nan")

    return {
        "oa": round(float(oa), 4),
        "miou": round(float(miou), 4),
        "kappa": round(float(kappa), 4),
        "f1_crop": round(float(f1_crop), 4),
        "iou_crop": round(float(iou_crop), 4) if not np.isnan(iou_crop) else None,
        "conf_matrix": cm.tolist(),
    }
