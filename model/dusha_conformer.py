import torch
import pytorch_lightning as pl
import torch.nn as nn
from collections import defaultdict
from torch.nn import functional as F

from .lightning_metrics import normed_cm_tensor, Metrics
from lion_pytorch import Lion


class EmotionClassifier(pl.LightningModule):
    def __init__(self, model, clf, lr=1e-5, device="cuda"):
        super().__init__()
        self.ssl_encoder = model
        self.clf = clf
        self.pooling = nn.AdaptiveAvgPool2d((8, 100))
        self.logsoftmax = torch.nn.LogSoftmax(-1)
        self.metrics = Metrics().to(device)
        self.plot_cm = True
        self.dataset_mapper = {0: "crowd", 1: "podcast"}
        self.lr = lr

    def forward(self, input_values, lenghts):
        logits = self.ssl_encoder(audio_signal=input_values, length=lenghts)[0]

        logprobs = self.logsoftmax(self.clf(self.pooling(logits)))

        return logprobs

    def training_step(self, batch, batch_idx):
        input_values, lenghts, labels = batch
        logprobs = self.forward(input_values, lenghts)
        loss = F.nll_loss(logprobs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        input_values, lenghts, labels = batch
        logprobs = self.forward(input_values, lenghts)
        loss = F.nll_loss(logprobs, labels)
        preds = logprobs.argmax(dim=-1)
        res_dict = defaultdict(float)
        res_dict["label"] = labels
        res_dict["pred"] = preds
        res_dict["val_loss"] = loss.item()
        res_dict["dataset_idx"] = dataset_idx

        return res_dict

    def validation_epoch_end(self, validation_step_outputs):
        if not isinstance(validation_step_outputs[0], list):
            validation_step_outputs = [validation_step_outputs]

        for i in range(len(validation_step_outputs)):
            res_dict = defaultdict(list)
            loss = 0
            for output in validation_step_outputs[i]:
                res_dict["label"] += [output["label"]]
                res_dict["pred"] += [output["pred"]]
                dataset_id = output["dataset_idx"]
                loss += output["val_loss"] / len(validation_step_outputs[i])
            loss = loss / len(validation_step_outputs)
            res_dict["label"] = torch.cat(res_dict["label"])
            res_dict["pred"] = torch.cat(res_dict["pred"])
            res_dict = self.all_gather(res_dict)
            res_dict["label"] = res_dict["label"].reshape(-1)
            res_dict["pred"] = res_dict["pred"].reshape(-1)
            metrics = self.metrics(
                res_dict["pred"], res_dict["label"], self.dataset_mapper[dataset_id]
            )
            if self.global_rank == 0:
                self.log_dict(metrics, sync_dist=False)
                if not self.trainer.sanity_checking and self.plot_cm:
                    tensorboard_logger = self.logger.experiment
                    tensorboard_logger.add_image(
                        f"Confusion matrix {self.dataset_mapper[dataset_id]}",
                        normed_cm_tensor(res_dict["label"], res_dict["pred"]),
                        self.current_epoch,
                    )
            self.log(
                f"val_loss_{self.dataset_mapper[dataset_id]}", loss, sync_dist=True
            )

    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), self.lr)

        return optimizer


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(800, 768)
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(768, 4)
        self.flatten = nn.Flatten()

    def forward(self, features):
        x = features
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
