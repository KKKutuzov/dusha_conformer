import numpy as np
import seaborn as sns
import tempfile

from torchvision.io import read_image
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
)
from typing import Dict, Optional

GLOBAL_METRICS = {
    "WA": MulticlassAccuracy(4, average="macro"),
    "weighted_f1": MulticlassF1Score(4, average="weighted"),
    "macro_f1": MulticlassF1Score(4, average="macro"),
    "UA": MulticlassAccuracy(4, average="micro"),
}

PER_CLASS_METRICS = {
    "recall": MulticlassRecall(4, average="none"),
    "precision": MulticlassPrecision(4, average="none"),
}


class Metrics:
    def __init__(
        self,
        global_metrics: Optional[Dict] = GLOBAL_METRICS,
        per_class_metrics: Optional[Dict] = PER_CLASS_METRICS,
    ):
        """
        global_metrics - Dict of (name_of_metric,  metric) with metrics that have only one value for all classes(WA, F1, etc.)
        per_class_metrics - Dict of (name_of_metric,  metric) with metrics that have  one value for each class(recall, precision)
        """
        self.global_metrics = global_metrics
        self.per_class_metrics = per_class_metrics

    def __call__(self, preds, labels, dataset_id):
        """
        returns Dict of (name, metric_value)
        """
        res_dict = {}
        for name, metric in self.global_metrics.items():
            res_dict[f"{name}_{dataset_id}"] = metric(preds, labels)

        for name, metric in self.per_class_metrics.items():
            values = metric(preds, labels)
            for i in range(len(values)):
                res_dict[f"{name}_{i}_{dataset_id}"] = values[i]

        return res_dict

    def to(self, device):
        for name, metric in self.global_metrics.items():
            self.global_metrics[name] = metric.to(device)

        for name, metric in self.per_class_metrics.items():
            self.per_class_metrics[name] = metric.to(device)
        return self


def normed_cm_tensor(gts, preds):
    with tempfile.NamedTemporaryFile() as tmp:
        normed_cm_to_file(gts, preds, tmp.name)
        image = read_image(tmp.name)
    return image


def normed_cm_to_file(gts, preds, filename):
    num_classes = gts.max() + 1
    num_preds = preds.max() + 1
    labels = list(range(num_classes))
    pred_labels = list(range(num_preds))
    CM = np.zeros((num_classes, num_preds))

    for i, value in enumerate(range(num_classes - 1, -1, -1)):
        for j, pred_label in enumerate(range(num_preds)):
            CM[i, j] = ((gts == value) & (preds == pred_label)).sum()
    CM_normed = CM.copy() / (CM.sum(axis=1)[:, None] + 1e-6)
    fig = sns.heatmap(
        CM_normed.round(2),
        xticklabels=pred_labels,
        yticklabels=labels[::-1],
        annot=True,
        cbar=True,
    )
    fig.set(xlabel="Pred", ylabel="GT")
    fig = fig.get_figure()
    fig.savefig(filename, format="png")
    fig.clear()
